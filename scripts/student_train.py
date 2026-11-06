"""Single approved teacher-clip overfit diagnostic, not a general avatar trainer."""
import argparse
import json
import math
import sys
import time
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def predict(model, image, audio):
    import torch
    from student.model import WIDTH
    outputs = []
    with torch.inference_mode():
        spatial, identity = model.encode_identity(image)
        state = torch.zeros(1, WIDTH, device=image.device)
        for row in audio:
            rgb, state = model.forward_frame(row[None], state, spatial, identity)
            if not torch.isfinite(rgb).all() or not torch.isfinite(state).all():
                raise RuntimeError('non-finite recurrent prediction')
            outputs.append(rgb[0].cpu())
    return torch.stack(outputs)


def metrics(prediction, target, box):
    x0, y0, x1, y1 = box
    mouth_prediction = prediction[:, :, y0:y1, x0:x1]
    mouth_target = target[:, :, y0:y1, x0:x1]
    edge_error = .5*((mouth_prediction[:, :, :, 1:]-mouth_prediction[:, :, :, :-1]
                     -mouth_target[:, :, :, 1:]+mouth_target[:, :, :, :-1]).abs().mean()
                    +(mouth_prediction[:, :, 1:]-mouth_prediction[:, :, :-1]
                      -mouth_target[:, :, 1:]+mouth_target[:, :, :-1]).abs().mean())
    return {'mae': float((prediction-target).abs().mean()),
            'mse': float((prediction-target).square().mean()),
            'mouth_roi_mae': float((prediction[:, :, y0:y1, x0:x1]-target[:, :, y0:y1, x0:x1]).abs().mean()),
            'mouth_edge_mae': float(edge_error),
            'mouth_temporal_delta_mae': float(((mouth_prediction[1:]-mouth_prediction[:-1])
                                               -(mouth_target[1:]-mouth_target[:-1])).abs().mean()),
            'temporal_delta_mae': float(((prediction[1:]-prediction[:-1])-(target[1:]-target[:-1])).abs().mean()),
            'predicted_frame_delta_mean': float((prediction[1:]-prediction[:-1]).abs().mean()),
            'teacher_frame_delta_mean': float((target[1:]-target[:-1]).abs().mean())}


def pixel_box(values, label):
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
        raise ValueError(f'{label} coordinates must be finite and in [0, 1]')
    box = tuple(round(value*256) for value in values)
    x0, y0, x1, y1 = box
    if x0 >= x1 or y0 >= y1:
        raise ValueError(f'{label} must define a nonempty normalized rectangle')
    return box


def write_video(path, frames, waveform):
    import av
    import numpy as np
    with av.open(str(path), 'w') as container:
        video = container.add_stream('libx264', rate=16)
        video.width, video.height = frames.shape[2], frames.shape[1]
        video.pix_fmt = 'yuv420p'; video.options = {'crf': '18', 'preset': 'veryfast'}
        audio = container.add_stream('aac', rate=16000); audio.layout = 'mono'
        for index, pixels in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(pixels, format='rgb24')
            frame.pts = index; frame.time_base = Fraction(1, 16)
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode(None):
            container.mux(packet)
        for start in range(0, len(waveform), 1024):
            samples = np.ascontiguousarray(waveform[start:start+1024][None], dtype=np.float32)
            frame = av.AudioFrame.from_ndarray(samples, format='fltp', layout='mono')
            frame.sample_rate = 16000; frame.pts = start; frame.time_base = Fraction(1, 16000)
            for packet in audio.encode(frame):
                container.mux(packet)
        for packet in audio.encode(None):
            container.mux(packet)


def train(args):
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from student.audio import CONTRACT, histories
    from student.data import load, sha256
    from student.model import ARCHITECTURE, Candidate, WIDTH

    architecture = getattr(args, 'architecture', ARCHITECTURE)
    if architecture == 'reference_warp_residual_v1':
        from student.reference_model import ReferenceCandidate
        model_type = ReferenceCandidate
    elif architecture == ARCHITECTURE:
        model_type = Candidate
    else:
        raise ValueError('unknown student architecture')

    torch.set_num_threads(2); torch.manual_seed(args.seed)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    record, data = load(args.dataset)
    if args.sequence_length > record['frames']:
        raise ValueError('sequence length exceeds source frames')
    box = pixel_box(args.mouth_box, 'mouth box')
    x0, y0, x1, y1 = box
    hand_boxes = [pixel_box(values, 'hand box') for values in getattr(args, 'hand_box', [])]
    model = model_type()
    initial_checkpoint_sha256 = None
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location='cpu', weights_only=True)
        if checkpoint.get('architecture') != architecture or checkpoint.get('schema_version') != 1:
            raise ValueError('initial checkpoint architecture mismatch')
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        initial_checkpoint_sha256 = sha256(args.init_checkpoint)
    else:
        for parameter in model.parameters():
            if parameter.ndim > 1:
                torch.nn.init.xavier_uniform_(parameter, gain=.9)
            else:
                torch.nn.init.uniform_(parameter, -.03, .03)
        if hasattr(model, 'reset_output_head'):
            model.reset_output_head()
    model = model.to(device)
    image = torch.from_numpy(data['reference'].copy()).permute(2, 0, 1)[None].float().to(device)/255
    target_cpu = torch.from_numpy(data['frames'].copy()).permute(0, 3, 1, 2).float()/255
    target = target_cpu.to(device)
    audio = torch.from_numpy(histories(data['audio_features'])).to(device)
    args.output.mkdir(parents=True, exist_ok=False)
    reference = Image.fromarray(data['reference'])
    ImageDraw.Draw(reference).rectangle(box, outline='yellow', width=2)
    reference.save(args.output / 'mouth_roi_review.png')
    if hand_boxes:
        for region in hand_boxes:
            ImageDraw.Draw(reference).rectangle(region, outline='cyan', width=2)
        reference.save(args.output / 'hand_roi_review.png')
    model.eval(); initial = metrics(predict(model, image, audio), target_cpu, box)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    log = []; started = time.perf_counter()
    for step in range(1, args.steps+1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        spatial, identity = model.encode_identity(image)
        state = torch.zeros(1, WIDTH, device=device)
        start = int(rng.integers(0, record['frames']-args.sequence_length+1))
        previous = None
        with torch.no_grad():
            for index in range(max(0, start-8), start):
                previous, state = model.forward_frame(audio[index:index+1], state, spatial, identity)
        reconstruction, mouth, temporal, hand = [], [], [], []
        for index in range(start, start+args.sequence_length):
            rgb, state = model.forward_frame(audio[index:index+1], state, spatial, identity)
            truth = target[index:index+1]
            difference = (rgb-truth).abs()
            reconstruction.append(difference.mean())
            mouth.append(difference[:, :, y0:y1, x0:x1].mean())
            for left, top, right, bottom in hand_boxes:
                hand.append(difference[:, :, top:bottom, left:right].mean())
            if previous is not None:
                temporal.append(((rgb-previous)-(truth-target[index-1:index])).abs().mean())
            previous = rgb
        reconstruction = torch.stack(reconstruction).mean(); mouth = torch.stack(mouth).mean()
        temporal = torch.stack(temporal).mean() if temporal else reconstruction*0
        loss = reconstruction + 3*mouth + .5*temporal
        hand_loss = torch.stack(hand).mean() if hand else None
        if hand_loss is not None:
            loss = loss + 2*hand_loss
        if not torch.isfinite(loss):
            raise RuntimeError('non-finite training loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 20 == 0 or step == args.steps:
            row = {'step': step, 'loss': float(loss.detach()), 'reconstruction': float(reconstruction.detach()),
                   'mouth_roi': float(mouth.detach()), 'temporal': float(temporal.detach()),
                   'gradient_norm_before_clip': float(norm), 'elapsed_seconds': time.perf_counter()-started}
            if hand_loss is not None:
                row['hand_roi'] = float(hand_loss.detach())
            log.append(row); print(json.dumps(row), flush=True)
    train_seconds = time.perf_counter()-started
    model.eval(); prediction = predict(model, image, audio)
    shuffled = torch.from_numpy(histories(np.roll(data['audio_features'], record['frames']//2, axis=0))).to(device)
    shuffled_prediction = predict(model, image, shuffled)
    silence = torch.full_like(audio, -1.)
    silence_prediction = predict(model, image, silence)
    final = metrics(prediction, target_cpu, box)
    shuffled_metrics = metrics(shuffled_prediction, target_cpu, box)
    static_mean = target_cpu.mean(0, keepdim=True).expand_as(target_cpu)
    report = {'schema_version': 1, 'purpose': 'single_clip_overfit_diagnostic', 'architecture': architecture,
              'device': device, 'torch_version': torch.__version__, 'steps': args.steps,
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'initial_checkpoint_sha256': initial_checkpoint_sha256, 'optimizer_state_restored': False,
              'sequence_length': args.sequence_length, 'learning_rate': args.lr, 'seed': args.seed,
              'loss_weights': {'reconstruction_l1': 1, 'manual_mouth_roi_l1': 3, 'temporal_difference_l1': .5},
              'mouth_box_pixels': box, 'mouth_box_type': 'manual_fixed_region_not_face_tracking',
              'dataset_manifest': str(args.dataset.resolve()), 'dataset_manifest_sha256': sha256(args.dataset),
              'source_video_sha256': record['source_video_sha256'], 'audio_contract': CONTRACT,
              'training_seconds': train_seconds, 'frames': record['frames'], 'fps': 16,
              'evaluation_split': 'same_training_clip_reconstruction', 'independent_validation': False,
              'lip_sync_verified': False, 'generalization_verified': False, 'video_quality_validated': False,
              'initial': initial, 'final': final, 'audio_shift_half_clip': shuffled_metrics,
              'static_training_mean_oracle': metrics(static_mean, target_cpu, box),
              'static_reference_copy': metrics(image[0].detach().cpu()[None].expand_as(target_cpu), target_cpu, box),
              'audio_shift_output_change_mae': float((prediction-shuffled_prediction).abs().mean()),
              'silence_output_change_mae': float((prediction-silence_prediction).abs().mean()), 'log': log}
    if hand_boxes:
        report['hand_boxes_pixels'] = hand_boxes
        report['loss_weights']['manual_hand_roi_l1'] = 2
    provenance = {'purpose': report['purpose'], 'teacher_trained': True, 'optimizer_steps': args.steps,
                  'dataset_manifest_sha256': report['dataset_manifest_sha256'],
                  'source_video_sha256': record['source_video_sha256'], 'audio_contract': CONTRACT,
                  'evaluation_split': report['evaluation_split'], 'video_quality_validated': False}
    torch.save({'schema_version': 1, 'architecture': architecture,
                'state_dict': {name: value.detach().cpu() for name, value in model.state_dict().items()},
                'provenance': provenance}, args.output / 'checkpoint.pt')
    pixels = prediction.permute(0, 2, 3, 1).numpy()
    pixels = np.rint(np.clip(pixels, 0, 1)*255).astype(np.uint8)
    write_video(args.output / 'student.mp4', pixels, data['audio'])
    write_video(args.output / 'comparison.mp4', np.concatenate([data['frames'], pixels], axis=2), data['audio'])
    grid = Image.new('RGB', (3*256, 2*280), 'white'); draw = ImageDraw.Draw(grid)
    for column, index in enumerate((0, record['frames']//2, record['frames']-1)):
        for row, (title, source) in enumerate((('Teacher', data['frames']), ('Student', pixels))):
            grid.paste(Image.fromarray(source[index]), (column*256, row*280+24))
            draw.text((column*256+6, row*280+5), f'{title} t={index/16:.2f}s', fill='black')
    grid.save(args.output / 'comparison.png')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'log'}, indent=2), flush=True)
    print(f'Report: {(args.output / "report.json").resolve()}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--init-checkpoint', type=Path)
    parser.add_argument('--architecture', choices=['recurrent_conditioned_rgb_v0', 'reference_warp_residual_v1'],
                        default='recurrent_conditioned_rgb_v0')
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--sequence-length', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=713)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--mouth-box', nargs=4, type=float, default=[.35, .36, .65, .52], metavar=('X0','Y0','X1','Y1'))
    parser.add_argument('--hand-box', nargs=4, type=float, action='append', default=[],
                        metavar=('X0','Y0','X1','Y1'),
                        help='repeatable normalized hand motion region for extra reconstruction supervision')
    args = parser.parse_args()
    if not 1 <= args.steps <= 10000 or not 2 <= args.sequence_length <= 16 or not math.isfinite(args.lr) or not 0 < args.lr <= .01:
        parser.error('steps 1..10000, sequence length 2..16, finite learning rate (0,.01] required')
    train(args)


if __name__ == '__main__':
    main()
