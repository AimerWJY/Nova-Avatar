"""Build a bounded full-sequence FP32 reference fixture for converted v2 graphs."""
import argparse
import json
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def prepare(checkpoint_path, dataset, model_manifest, output, frames=None):
    import numpy as np
    import torch
    from student.audio import histories
    from student.data import load, sha256
    from student.mouth_model import MouthCandidate, ARCHITECTURE
    from student.mobile import fixed_letterbox_mask

    torch.set_num_threads(2)
    export = json.loads(model_manifest.read_text())
    if (export.get('architecture') != ARCHITECTURE or not export.get('ncnn_conversion', {}).get('passed')
            or export.get('checkpoint_sha256') != sha256(checkpoint_path)
            or export.get('dataset_manifest_sha256') != sha256(dataset)):
        raise ValueError('converted model, checkpoint and dataset do not match')
    record, data = load(dataset)
    mask = None
    policy = export.get('mobile_output_policy', {'version': 'raw_v2'})
    if policy['version'] == 'fixed_letterbox_mask_v1':
        mask, verified_policy = fixed_letterbox_mask(record, data)
        if verified_policy != policy:
            raise ValueError('exported letterbox geometry no longer matches source')
    elif policy['version'] != 'raw_v2':
        raise ValueError('unsupported output policy')
    count = len(data['audio_features']) if frames is None else frames
    if type(count) is not int or not 8 <= count <= len(data['audio_features']):
        raise ValueError('fixture frames must be 8..source frame count')
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
    if checkpoint.get('architecture') != ARCHITECTURE:
        raise ValueError('v2 checkpoint required')
    model = MouthCandidate(); model.load_state_dict(checkpoint['state_dict'], strict=True); model.eval()
    image = torch.from_numpy(data['reference'].copy()).permute(2, 0, 1)[None].float()/255
    selected_features = data['audio_features'][:count]
    audio = torch.from_numpy(histories(selected_features))
    padded_features = np.concatenate((np.repeat(selected_features[:1], 7, axis=0), selected_features))
    output.mkdir(parents=True, exist_ok=False)
    fixture = output/'v2.fixture'
    def write(stream, tensor):
        if not torch.isfinite(tensor).all():
            raise ValueError('non-finite reference fixture')
        stream.write(tensor.detach().contiguous().numpy().astype('<f4', copy=False).tobytes())
    with fixture.open('xb') as stream, torch.inference_mode():
        stream.write(b'NOVAV202'); stream.write(struct.pack('<II', 2, len(audio)))
        write(stream, image); write(stream, torch.from_numpy(padded_features))
        cache, identity = model.encode_identity(image)
        write(stream, cache[0]); write(stream, identity)
        state = torch.zeros(1, 128)
        for row in audio:
            rgb, state = model.forward_frame(row[None], state, cache, identity)
            if mask is not None:
                rgb = rgb*mask
            write(stream, rgb); write(stream, state)
    report = {'schema_version': 1, 'format': 'converted_v2_fixture_v1', 'architecture': ARCHITECTURE,
              'model_manifest_sha256': sha256(model_manifest), 'checkpoint_sha256': sha256(checkpoint_path),
              'dataset_manifest_sha256': sha256(dataset), 'fixture_file': fixture.name,
              'fixture_sha256': sha256(fixture), 'fixture_bytes': fixture.stat().st_size,
              'frames': len(audio), 'source': 'prepared_training_clip',
              'mobile_output_policy': policy,
              'has_audio_frontend_on_phone': False, 'video_quality_validated': False}
    (output/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--model-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, help='optional prefix for a smaller diagnostic fixture')
    args = parser.parse_args()
    prepare(args.checkpoint, args.dataset, args.model_manifest, args.output, args.frames)


if __name__ == '__main__':
    main()
