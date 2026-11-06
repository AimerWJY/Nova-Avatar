"""Trace/freeze the actual v2 model, replay parity, optionally attempt pinned pnnx conversion."""
import argparse
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def export(checkpoint_path, dataset, output, convert_ncnn=False, preserve_letterbox=False, gpu_attention=False):
    import torch
    from student.audio import histories
    from student.data import load, sha256
    from student.mobile import IdentityEncoder, FrameDecoder, fixed_letterbox_mask
    from student.mouth_model import MouthCandidate, ARCHITECTURE

    torch.set_num_threads(2)
    record, data = load(dataset)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if (checkpoint.get('architecture') != ARCHITECTURE or checkpoint.get('schema_version') != 1
            or checkpoint.get('provenance', {}).get('dataset_manifest_sha256') != sha256(dataset)):
        raise ValueError('v2 checkpoint and exact prepared dataset are required')
    model = MouthCandidate(); model.load_state_dict(checkpoint['state_dict'], strict=True); model.eval()
    image = torch.from_numpy(data['reference'].copy()).permute(2, 0, 1)[None].float()/255
    audio = torch.from_numpy(histories(data['audio_features']))
    zero = torch.zeros(1, 128)
    content_mask, policy = None, {'version': 'raw_v2'}
    if preserve_letterbox:
        content_mask, policy = fixed_letterbox_mask(record, data)
    encoder, decoder = IdentityEncoder(model).eval(), FrameDecoder(model, content_mask).eval()
    with torch.no_grad():
        spatial, identity = encoder(image)
        traced_encoder = torch.jit.freeze(torch.jit.trace(encoder, (image,), check_trace=True).eval())
        inputs = (image, spatial, identity, audio[:1], zero)
        alternate = (image, spatial, identity, audio[1:2], torch.full_like(zero, .125))
        traced_decoder = torch.jit.freeze(torch.jit.trace(decoder, inputs, check_trace=True,
                                                       check_inputs=[alternate]).eval())
    output.mkdir(parents=True, exist_ok=False)
    encoder_path, decoder_path = output/'encoder.pt', output/'frame.pt'
    traced_encoder.save(str(encoder_path)); traced_decoder.save(str(decoder_path))
    loaded_encoder, loaded_decoder = torch.jit.load(str(encoder_path)), torch.jit.load(str(decoder_path))
    max_abs, elements, max_output_change, max_content_change = 0., 0, 0., 0.

    def check(actual, expected):
        nonlocal max_abs, elements
        if not torch.isfinite(actual).all():
            raise ValueError('non-finite serialized TorchScript output')
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        max_abs = max(max_abs, float((actual-expected).abs().max())); elements += actual.numel()

    with torch.inference_mode():
        ts_spatial, ts_identity = loaded_encoder(image)
        check(ts_spatial, spatial); check(ts_identity, identity)
        state, ts_state = zero.clone(), zero.clone()
        cache, expected_identity = model.encode_identity(image)
        for row in audio:
            expected, state = model.forward_frame(row[None], state, cache, expected_identity)
            if content_mask is not None:
                updated = expected*content_mask
                max_output_change = max(max_output_change, float((updated-expected).abs().max()))
                max_content_change = max(max_content_change, float(((updated-expected)*content_mask).abs().max()))
                expected = updated
            actual, ts_state = loaded_decoder(image, ts_spatial, ts_identity, row[None], ts_state)
            check(actual, expected); check(ts_state, state)
    report = {'schema_version': 1, 'architecture': ARCHITECTURE,
              'checkpoint_sha256': sha256(checkpoint_path), 'dataset_manifest_sha256': sha256(dataset),
              'torch_version': torch.__version__, 'torchscript_parity_passed': True,
              'reference_frames': len(audio), 'checked_elements': elements, 'max_abs_error': max_abs,
              'fixed_batch_size': 1, 'image_size': [256, 256], 'audio_history_shape': [8, 40],
              'mobile_output_policy': policy, 'max_change_from_raw_output': max_output_change,
              'max_change_in_content_region': max_content_change,
              'encoder_inputs': ['image'], 'encoder_outputs': ['spatial', 'identity'],
              'frame_inputs': ['image', 'spatial', 'identity', 'audio', 'state'], 'frame_outputs': ['rgb', 'next_state'],
              'artifacts': {path.name: {'sha256': sha256(path), 'bytes': path.stat().st_size}
                            for path in (encoder_path, decoder_path)},
              'ncnn_conversion': {'requested': convert_ncnn, 'passed': False, 'runtime_load_tested': False},
              'phone_validated': False, 'lip_sync_verified': False, 'video_quality_validated': False}
    failure = None
    if convert_ncnn:
        try:
            version = importlib.metadata.version('pnnx')
            if version != '20261020':
                raise ValueError('pnnx==20261020 required; no automatic environment replacement')
            tool = Path(sys.executable).parent/'pnnx'
            if not tool.is_file():
                found = shutil.which('pnnx')
                if not found:
                    raise ValueError('pnnx CLI not found')
                tool = Path(found)
            report['ncnn_conversion'].update(version=version, graphs=[])
            shapes = {'encoder': '[1,3,256,256]',
                      'frame': '[1,3,256,256],[1,32,64,64],[1,128],[1,8,40],[1,128]'}
            for name in ('encoder', 'frame'):
                command = [str(tool.resolve()), name+'.pt', 'inputshape='+shapes[name], 'fp16=0']
                print('Converting '+name, flush=True)
                result = subprocess.run(command, cwd=output, capture_output=True, text=True, timeout=180)
                (output/(name+'.conversion.log')).write_text(result.stdout+'\n'+result.stderr)
                param, binary = output/(name+'.ncnn.param'), output/(name+'.ncnn.bin')
                if result.returncode or not param.is_file() or not binary.is_file():
                    raise ValueError(f'{name} conversion failed; see {name}.conversion.log')
                rewrites = []
                if gpu_attention and name == 'frame':
                    from scripts.ncnn_v2_rewrite import split_attention_gemm
                    rewrites = split_attention_gemm(param)
                lines = param.read_text().splitlines()
                if not lines or lines[0].strip() != '7767517':
                    raise ValueError('invalid ncnn parameter magic')
                layer_types = [line.split()[0] for line in lines[2:] if line.strip()]
                unsupported = sorted({kind for kind in layer_types if kind.startswith(('pnnx.', 'aten', 'prim', 'Tensor.', 'F.'))})
                report['ncnn_conversion']['graphs'].append({'name': name, 'layer_types': sorted(set(layer_types)),
                                                            'attention_rewrites': rewrites,
                                                            'unconverted_types': unsupported,
                                                            'param_sha256': sha256(param), 'bin_sha256': sha256(binary)})
                if unsupported:
                    raise ValueError(f'{name} contains unconverted operators: {unsupported}')
            report['ncnn_conversion']['passed'] = True
        except (OSError, ValueError, subprocess.TimeoutExpired, importlib.metadata.PackageNotFoundError) as exc:
            failure = str(exc); report['ncnn_conversion']['error'] = failure
    (output/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    if failure:
        raise RuntimeError(failure)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--convert-ncnn', action='store_true')
    parser.add_argument('--preserve-letterbox', action='store_true', help='fixed geometry: keep verified source/teacher zero padding black')
    parser.add_argument('--gpu-attention', action='store_true', help='fixed v2: split batched attention MatMul into 2D Gemm nodes')
    args = parser.parse_args()
    if args.gpu_attention and not args.convert_ncnn:
        parser.error('--gpu-attention requires --convert-ncnn')
    export(args.checkpoint, args.dataset, args.output, args.convert_ncnn, args.preserve_letterbox, args.gpu_attention)


if __name__ == '__main__':
    main()
