"""Fit a local mouth decoder with balanced dark-area sampling and teacher supervision."""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'scripts'))


def train(args):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from PIL import Image, ImageDraw
    from student.audio import CONTRACT, histories
    from student.data import load, sha256
    from student.model import WIDTH
    from student.reference_model import ARCHITECTURE as BASE_ARCHITECTURE
    from student.mouth_model import ARCHITECTURE, MouthCandidate, CORE_BOX, PATCH_BOX, mouth_luma, dark_area_proxy
    from student_train import predict, metrics, write_video

    torch.set_num_threads(2); torch.manual_seed(args.seed)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    record, data = load(args.dataset)
    checkpoint = torch.load(args.base_checkpoint, map_location='cpu', weights_only=True)
    if (checkpoint.get('architecture') != BASE_ARCHITECTURE or checkpoint.get('schema_version') != 1
            or checkpoint.get('provenance', {}).get('dataset_manifest_sha256') != sha256(args.dataset)):
        raise ValueError('requires the v1 checkpoint fitted on this exact prepared dataset')
    model = MouthCandidate()
    missing = model.load_state_dict(checkpoint['state_dict'], strict=False)
    if missing.unexpected_keys or set(missing.missing_keys) != {name for name in model.state_dict() if name.startswith('mouth.')}:
        raise ValueError('unexpected base/mouth checkpoint mapping')
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith('mouth.'))
    model = model.to(device)
    reference = torch.from_numpy(data['reference'].copy()).permute(2, 0, 1)[None].float().to(device)/255
    truth_cpu = torch.from_numpy(data['frames'].copy()).permute(0, 3, 1, 2).float()/255
    truth = truth_cpu.to(device)
    audio = torch.from_numpy(histories(data['audio_features'])).to(device)
    model.eval(); base_frames, base_states = [], []
    with torch.inference_mode():
        cache, identity = model.encode_identity(reference); state = torch.zeros(1, WIDTH, device=device)
        for row in audio:
            rgb, state = model.forward_base_frame(row[None], state, cache, identity)
            base_frames.append(rgb[0]); base_states.append(state[0])
    # Normal tensors allow training the new head from the cached frozen backbone.
    base = torch.stack(base_frames).clone(); states = torch.stack(base_states).clone()
    target_scores = dark_area_proxy(truth).detach()
    scores_np = target_scores.cpu().numpy()
    open_group = np.flatnonzero(scores_np >= np.quantile(scores_np, .75))
    closed_group = np.flatnonzero(scores_np <= np.quantile(scores_np, .25))
    args.output.mkdir(parents=True, exist_ok=False)
    optimizer = torch.optim.Adam(model.mouth.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed); log = []; start_time = time.perf_counter()
    x0, y0, x1, y1 = CORE_BOX
    for step in range(1, args.steps+1):
        model.mouth.train(); optimizer.zero_grad(set_to_none=True)
        current = np.concatenate([rng.choice(open_group, args.batch_size//2),
                                  rng.choice(closed_group, args.batch_size-args.batch_size//2)])
        previous = np.maximum(current-1, 0)
        indices = torch.as_tensor(np.concatenate([current, previous]), device=device)
        rgb = model.mouth(base[indices], reference.expand(len(indices), -1, -1, -1), audio[indices], states[indices])
        cropped = rgb[:, :, y0:y1, x0:x1]; target_crop = truth[indices, :, y0:y1, x0:x1]
        pixel = (cropped-target_crop).abs().mean()
        edge = .5*((cropped[:, :, 1:]-cropped[:, :, :-1]-target_crop[:, :, 1:]+target_crop[:, :, :-1]).abs().mean()
                   +(cropped[:, :, :, 1:]-cropped[:, :, :, :-1]-target_crop[:, :, :, 1:]+target_crop[:, :, :, :-1]).abs().mean())
        dark_target = ((.22-mouth_luma(truth[indices]))/.04).sigmoid().detach()
        dark_logits = (.22-mouth_luma(rgb))/.04
        dark = F.binary_cross_entropy_with_logits(dark_logits, dark_target)
        area = (dark_area_proxy(rgb)-target_scores[indices]).abs().mean()
        n = args.batch_size
        temporal = ((cropped[:n]-cropped[n:])-(target_crop[:n]-target_crop[n:])).abs().mean()
        loss = pixel+.5*edge+.1*dark+area+.5*temporal
        if not torch.isfinite(loss):
            raise RuntimeError('non-finite mouth loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.mouth.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            row = {'step': step, 'loss': float(loss.detach()), 'pixel': float(pixel.detach()),
                   'dark_bce': float(dark.detach()), 'dark_area': float(area.detach()),
                   'edge': float(edge.detach()), 'temporal': float(temporal.detach()),
                   'elapsed_seconds': time.perf_counter()-start_time}
            log.append(row); print(json.dumps(row), flush=True)
    elapsed = time.perf_counter()-start_time
    model.eval(); corrected = predict(model, reference, audio)
    shifted_audio = torch.from_numpy(histories(np.roll(data['audio_features'], len(audio)//2, axis=0))).to(device)
    shifted = predict(model, reference, shifted_audio)
    base_cpu = base.detach().cpu()
    current_state = model.state_dict()
    if any(not torch.equal(current_state[name].cpu(), tensor) for name, tensor in checkpoint['state_dict'].items()):
        raise RuntimeError('frozen base weights changed')
    outside = torch.ones((256, 256), dtype=torch.bool)
    px0, py0, px1, py1 = PATCH_BOX; outside[py0:py1, px0:px1] = False
    outside_error = float((corrected-base_cpu)[:, :, outside].abs().max())
    if outside_error != 0:
        raise RuntimeError('mouth correction modified pixels outside its patch')

    def opening_summary(prediction):
        scores = dark_area_proxy(prediction)
        target = target_scores.cpu()
        corr = float(torch.corrcoef(torch.stack((target, scores)))[0, 1])
        return {'dark_area_mae': float((scores-target).abs().mean()),
                'dark_area_min': float(scores.min()), 'dark_area_max': float(scores.max()),
                'dark_area_std': float(scores.std()), 'teacher_proxy_correlation': corr if np.isfinite(corr) else None,
                'scores': scores.tolist()}

    report = {'schema_version': 1, 'architecture': ARCHITECTURE, 'purpose': 'single_clip_mouth_overfit',
              'steps': args.steps, 'batch_size': args.batch_size, 'learning_rate': args.lr, 'seed': args.seed,
              'device': device, 'training_seconds': elapsed, 'frames': len(truth), 'fps': 16,
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'trainable_parameter_count': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'base_checkpoint_sha256': sha256(args.base_checkpoint), 'dataset_manifest_sha256': sha256(args.dataset),
              'source_video_sha256': record['source_video_sha256'], 'base_frozen_and_unchanged': True,
              'outside_patch_max_error': outside_error, 'patch_box': PATCH_BOX, 'core_box': CORE_BOX,
              'loss_weights': {'core_l1': 1, 'core_edge': .5, 'soft_dark_bce': .1, 'dark_area_l1': 1, 'core_temporal': .5},
              'sampling': 'equal upper/lower teacher dark-area quartiles; each sampled frame paired with its predecessor',
              'opening_proxy': 'mean sigmoid((0.22-luma)/0.04) in fixed core; not landmark lip aperture or a sync score',
              'evaluation_split': 'same_training_clip_reconstruction', 'independent_validation': False,
              'lip_sync_verified': False, 'generalization_verified': False, 'phone_validated': False,
              'video_quality_validated': False,
              'baseline': metrics(base_cpu, truth_cpu, CORE_BOX), 'corrected': metrics(corrected, truth_cpu, CORE_BOX),
              'shifted_audio': metrics(shifted, truth_cpu, CORE_BOX),
              'teacher_opening': opening_summary(truth_cpu), 'baseline_opening': opening_summary(base_cpu),
              'corrected_opening': opening_summary(corrected), 'log': log}
    provenance = {'purpose': report['purpose'], 'teacher_trained': True, 'optimizer_steps': args.steps,
                  'dataset_manifest_sha256': report['dataset_manifest_sha256'], 'source_video_sha256': record['source_video_sha256'],
                  'base_checkpoint_sha256': report['base_checkpoint_sha256'], 'audio_contract': CONTRACT,
                  'evaluation_split': report['evaluation_split'], 'video_quality_validated': False}
    torch.save({'schema_version': 1, 'architecture': ARCHITECTURE,
                'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'provenance': provenance}, args.output / 'checkpoint.pt')
    def pixels(tensor):
        return np.rint(tensor.permute(0, 2, 3, 1).numpy().clip(0, 1)*255).astype(np.uint8)
    arrays = [data['frames'], pixels(base_cpu), pixels(corrected)]
    write_video(args.output / 'teacher_v1_v2.mp4', np.concatenate(arrays, axis=2), data['audio'])
    write_video(args.output / 'student.mp4', arrays[2], data['audio'])
    zoom_frames = []
    for index in range(len(truth)):
        image = Image.new('RGB', (600, 152), 'white'); draw = ImageDraw.Draw(image)
        for column, (title, array) in enumerate(zip(('Teacher', 'v1 before', 'v2 corrected'), arrays)):
            crop = Image.fromarray(array[index]).crop(PATCH_BOX).resize((200, 128), Image.Resampling.BILINEAR)
            image.paste(crop, (column*200, 24)); draw.text((column*200+5, 5), title, fill='black')
        zoom_frames.append(np.array(image))
    write_video(args.output / 'mouth_zoom.mp4', np.stack(zoom_frames), data['audio'])
    indices = [i for i in (0, 12, 20, 24, 39, 68, 72, 76) if i < len(truth)]
    sheet = Image.new('RGB', (200*len(indices), 3*152), 'white'); draw = ImageDraw.Draw(sheet)
    for column, index in enumerate(indices):
        for row, (title, array) in enumerate(zip(('Teacher', 'v1 before', 'v2 corrected'), arrays)):
            crop = Image.fromarray(array[index]).crop(PATCH_BOX).resize((200, 128), Image.Resampling.BILINEAR)
            sheet.paste(crop, (column*200, row*152+24))
            draw.text((column*200+4, row*152+5), f'{title} frame={index}', fill='black')
    sheet.save(args.output / 'mouths.jpg', quality=95)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('log', 'teacher_opening', 'baseline_opening', 'corrected_opening')}, indent=2))
    print('Opening diagnostic:', json.dumps({k: {a: b for a, b in report[k].items() if a != 'scores'}
          for k in ('teacher_opening', 'baseline_opening', 'corrected_opening')}))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--base-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=713)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    if not 1 <= args.steps <= 10000 or not 2 <= args.batch_size <= 32 or not 0 < args.lr <= .01:
        parser.error('steps 1..10000, batch size 2..32, learning rate (0,.01] required')
    train(args)


if __name__ == '__main__':
    main()
