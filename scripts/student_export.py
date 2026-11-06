"""Export candidate weights and PyTorch FP32 reference tensors for Android parity."""
import argparse
import hashlib
import json
import pickle
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def export(output, checkpoint=None, frames=160, smoke_train=False):
    import numpy as np
    import torch
    from student.model import ARCHITECTURE, Candidate, SIDE, WIDTH, HISTORY, MEL

    torch.set_num_threads(2)
    torch.manual_seed(713)
    model = Candidate().cpu().float()
    provenance = {'purpose': 'random_weight_parity_check', 'teacher_trained': False}
    if checkpoint:
        record = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if (not isinstance(record, dict) or record.get('architecture') != ARCHITECTURE
                or record.get('schema_version') != 1 or not isinstance(record.get('provenance'), dict)):
            raise ValueError('export currently supports recurrent_conditioned_rgb_v0 only; architecture/schema mismatch')
        model.load_state_dict(record['state_dict'], strict=True)
        provenance = record['provenance']
    else:
        # Nontrivial activations make ordering/shape errors visible in parity tests.
        for parameter in model.parameters():
            if parameter.ndim > 1:
                torch.nn.init.xavier_uniform_(parameter, gain=.9)
            else:
                torch.nn.init.uniform_(parameter, -.03, .03)
    if any(not torch.isfinite(value).all() for value in model.parameters()):
        raise ValueError('checkpoint contains non-finite parameters')
    generator = torch.Generator().manual_seed(927)
    image = torch.rand(1, 3, SIDE, SIDE, generator=generator)
    history = torch.randn(frames + HISTORY - 1, MEL, generator=generator) * .5
    gradient_report = None
    if smoke_train:
        if checkpoint:
            raise ValueError('--checkpoint and --smoke-train cannot be combined')
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        state = torch.zeros(1, WIDTH)
        spatial, identity = model.encode_identity(image)
        loss = torch.zeros(())
        for index in range(2):
            rgb, state = model.forward_frame(history[index:index + HISTORY][None], state, spatial, identity)
            # Synthetic targets only test BPTT/optimizer wiring, never avatar quality.
            loss = loss + (rgb - image.roll(index + 1, dims=-1)).square().mean() + .1 * state.square().mean()
        loss.backward()
        gradients = {name: bool(value.grad is not None and torch.isfinite(value.grad).all())
                     for name, value in model.named_parameters()}
        if not all(gradients.values()) or not torch.isfinite(loss):
            raise RuntimeError('gradient smoke check failed')
        optimizer.step()
        changed = [name for name, value in model.named_parameters()
                   if not torch.equal(before[name], value.detach())]
        if not changed:
            raise RuntimeError('optimizer did not update any parameter')
        gradient_report = {'loss': float(loss.detach()), 'unrolled_frames': 2,
                           'all_parameter_gradients_finite': True, 'updated_parameter_tensors': len(changed)}
        provenance = {'purpose': 'synthetic_gradient_check', 'teacher_trained': False, 'optimizer_steps': 1}
    model.eval()
    # Exclusive output directory prevents accidentally replacing exported training assets.
    output.mkdir(parents=True, exist_ok=False)
    torch.save({'schema_version': 1, 'architecture': ARCHITECTURE,
                'state_dict': model.state_dict(), 'provenance': provenance}, output / 'checkpoint.pt')
    bundle = output / 'candidate.bundle'
    parameter_records = []

    def write_tensor(stream, tensor):
        stream.write(tensor.detach().cpu().contiguous().numpy().astype('<f4', copy=False).tobytes())

    with bundle.open('wb') as stream, torch.inference_mode():
        weights = model.ncnn_weights()
        stream.write(b'NOVAAV01')
        stream.write(struct.pack('<II', 1, len(weights)))
        for node, values in enumerate(weights):
            stream.write(struct.pack('<I', len(values)))
            for slot, value in enumerate(values):
                stream.write(struct.pack('<I', value.numel()))
                write_tensor(stream, value)
                parameter_records.append({'node': node, 'slot': slot, 'shape': list(value.shape), 'elements': value.numel()})
        stream.write(struct.pack('<II', frames, 8))
        write_tensor(stream, image)
        write_tensor(stream, history)
        spatial, identity = model.encode_identity(image)
        write_tensor(stream, spatial)
        write_tensor(stream, identity)
        state = torch.zeros(1, WIDTH)
        rgb_min, rgb_max = 1., 0.
        for index in range(8):
            rgb, state = model.forward_frame(history[index:index + HISTORY][None], state, spatial, identity)
            if not torch.isfinite(rgb).all() or not torch.isfinite(state).all():
                raise ValueError('non-finite PyTorch reference output')
            rgb_min, rgb_max = min(rgb_min, float(rgb.min())), max(rgb_max, float(rgb.max()))
            write_tensor(stream, rgb)
            write_tensor(stream, state)
    record = {'schema_version': 1, 'format': 'nova_candidate_fixture_v1', 'architecture': ARCHITECTURE,
              'bundle_file': bundle.name, 'bundle_sha256': hashlib.sha256(bundle.read_bytes()).hexdigest(),
              'bundle_bytes': bundle.stat().st_size, 'frames': frames, 'reference_frames': 8,
              'parameter_count': sum(p.numel() for p in model.parameters()), 'weights': parameter_records,
              'torch_version': torch.__version__, 'numpy_version': np.__version__,
              'provenance': provenance, 'gradient_check': gradient_report,
              'reference_rgb_min': rgb_min, 'reference_rgb_max': rgb_max,
              'synthetic_inputs': True, 'video_quality_validated': False}
    (output / 'manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({key: value for key, value in record.items() if key != 'weights'}, indent=2))
    print(f'Manifest: {(output / "manifest.json").resolve()}')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='new export directory')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--frames', type=int, default=160)
    parser.add_argument('--smoke-train', action='store_true', help='one optimizer step with two synthetic frames; no distillation')
    args = parser.parse_args()
    if not 8 <= args.frames <= 160:
        parser.error('--frames must be 8..160')
    if args.checkpoint and args.smoke_train:
        parser.error('--checkpoint and --smoke-train cannot be combined')
    try:
        export(args.output, args.checkpoint, args.frames, args.smoke_train)
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, pickle.UnpicklingError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
