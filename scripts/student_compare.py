"""Replay v0/v1 checkpoints on the same training clip, including static controls."""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))


def compare(dataset, v0, v1, output):
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from student.audio import histories
    from student.data import load, sha256
    from student.model import Candidate, ARCHITECTURE
    from student.reference_model import ReferenceCandidate, ARCHITECTURE as REFERENCE_ARCHITECTURE
    from student_train import predict, metrics, write_video

    torch.set_num_threads(2)
    record, data = load(dataset)
    image = torch.from_numpy(data['reference'].copy()).permute(2, 0, 1)[None].float()/255
    target = torch.from_numpy(data['frames'].copy()).permute(0, 3, 1, 2).float()/255
    audio = torch.from_numpy(histories(data['audio_features']))
    shifted = torch.from_numpy(histories(np.roll(data['audio_features'], len(target)//2, axis=0)))
    box = (90, 92, 166, 133)
    report = {'schema_version': 1, 'purpose': 'same_training_clip_architecture_comparison',
              'dataset_manifest_sha256': sha256(dataset), 'frames': len(target), 'fps': 16,
              'independent_validation': False, 'lip_sync_verified': False, 'video_quality_validated': False,
              'phone_v1_validated': False, 'mouth_box_pixels': box,
              'static_reference_copy': metrics(image.expand_as(target), target, box),
              'static_training_mean_oracle': metrics(target.mean(0, keepdim=True).expand_as(target), target, box)}
    predictions = []
    for label, path, constructor, architecture in (
            ('v0', v0, Candidate, ARCHITECTURE), ('v1', v1, ReferenceCandidate, REFERENCE_ARCHITECTURE)):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if (checkpoint.get('architecture') != architecture or checkpoint.get('schema_version') != 1
                or checkpoint.get('provenance', {}).get('source_video_sha256') != record['source_video_sha256']
                or checkpoint.get('provenance', {}).get('dataset_manifest_sha256') != sha256(dataset)):
            raise ValueError('comparison requires matching architectures and the same teacher source/prepared dataset')
        model = constructor(); model.load_state_dict(checkpoint['state_dict'], strict=True); model.eval()
        prediction = predict(model, image, audio)
        shifted_prediction = predict(model, image, shifted)
        report[label] = {'architecture': architecture, 'checkpoint_sha256': sha256(path),
                         'parameter_count': sum(p.numel() for p in model.parameters()),
                         'normal_audio': metrics(prediction, target, box),
                         'shifted_audio': metrics(shifted_prediction, target, box),
                         'audio_shift_output_change_mae': float((prediction-shifted_prediction).abs().mean())}
        pixels = np.rint(prediction.permute(0, 2, 3, 1).numpy().clip(0, 1)*255).astype(np.uint8)
        predictions.append(pixels)
    output.mkdir(parents=True, exist_ok=False)
    combined = np.concatenate([data['frames'], *predictions], axis=2)
    write_video(output / 'teacher_v0_v1.mp4', combined, data['audio'])
    grid = Image.new('RGB', (3*256, 3*280), 'white'); draw = ImageDraw.Draw(grid)
    for column, index in enumerate((0, len(target)//2, len(target)-1)):
        for row, (title, pixels) in enumerate(zip(('Teacher', 'v0', 'v1'), [data['frames'], *predictions])):
            grid.paste(Image.fromarray(pixels[index]), (column*256, row*280+24))
            draw.text((column*256+6, row*280+5), f'{title} t={index/16:.2f}s', fill='black')
    grid.save(output / 'teacher_v0_v1.jpg', quality=95)
    (output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--v0-checkpoint', type=Path, required=True)
    parser.add_argument('--v1-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    compare(args.dataset, args.v0_checkpoint, args.v1_checkpoint, args.output)


if __name__ == '__main__':
    main()
