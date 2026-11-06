"""Run the v2 checkpoint on an independent audio source; no teacher frames used."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'scripts'))


def infer(args):
    import av
    import numpy as np
    import torch
    from PIL import Image
    from student.audio import CONTRACT, features, histories
    from student.data import audio_timeline, square_image, load, sha256
    from student.mouth_model import MouthCandidate, ARCHITECTURE, dark_area_proxy
    from student_train import predict, write_video

    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint.get('schema_version') != 1 or checkpoint.get('architecture') != ARCHITECTURE:
        raise ValueError('this inference entry point requires a mouth_refinement_v2 checkpoint')
    _, training = load(args.training_manifest)
    if checkpoint['provenance'].get('dataset_manifest_sha256') != sha256(args.training_manifest):
        raise ValueError('training manifest does not match checkpoint provenance')
    with av.open(str(args.audio_source)) as container:
        if not container.streams.audio:
            raise ValueError('source contains no audio stream')
        stream = container.streams.audio[0]
        duration = float(stream.duration*stream.time_base) if stream.duration is not None else container.duration/av.time_base
        first = next(container.decode(audio=0))
        if first.pts is None or first.time_base is None:
            raise ValueError('audio timestamps missing')
        start = float(first.pts*first.time_base)
    frames = min(int(duration*16), int(args.max_seconds*16))
    if frames < 8:
        raise ValueError('at least 0.5 seconds of audio is required')
    waveform, alignment = audio_timeline(args.audio_source, start, frames*1000)
    frames = min(frames, alignment['contiguous_audio_samples']//1000)
    if frames < 8:
        raise ValueError('not enough continuously covered audio')
    waveform = waveform[:frames*1000]
    common_length = min(len(waveform), len(training['audio']))
    identical_training_prefix = np.array_equal(waveform[:common_length], training['audio'][:common_length])
    if identical_training_prefix:
        raise ValueError('audio duplicates the training clip prefix; use an independent source')
    with Image.open(args.reference) as image:
        normalized_reference = square_image(image)
    image = torch.from_numpy(normalized_reference).permute(2, 0, 1)[None].float()/255
    audio_features = features(waveform, np.arange(frames)/16)
    audio = torch.from_numpy(histories(audio_features))
    model = MouthCandidate(); model.load_state_dict(checkpoint['state_dict'], strict=True); model.eval()
    prediction = predict(model, image, audio)
    silence_prediction = predict(model, image, torch.full_like(audio, -1.))
    if not torch.isfinite(prediction).all() or not torch.isfinite(silence_prediction).all():
        raise RuntimeError('non-finite prediction on independent input')
    args.output.mkdir(parents=True, exist_ok=False)
    def pixels(value):
        return np.rint(value.permute(0, 2, 3, 1).numpy().clip(0, 1)*255).astype(np.uint8)
    normal, silence = pixels(prediction), pixels(silence_prediction)
    write_video(args.output / 'student.mp4', normal, waveform)
    write_video(args.output / 'audio_vs_silence.mp4', np.concatenate((normal, silence), axis=2), waveform)
    Image.fromarray(normalized_reference).save(args.output / 'reference.png')
    indices = np.linspace(0, frames-1, 6, dtype=int)
    sheet = np.concatenate([normal[index] for index in indices], axis=1)
    Image.fromarray(sheet).save(args.output / 'frames.jpg', quality=95)
    scores = dark_area_proxy(prediction)
    report = {'schema_version': 1, 'purpose': 'independent_audio_preview_without_matched_teacher',
              'architecture': ARCHITECTURE, 'checkpoint_sha256': sha256(args.checkpoint),
              'source_audio_file': str(args.audio_source.resolve()), 'source_audio_file_sha256': sha256(args.audio_source),
              'decoded_audio_sha256': hashlib.sha256(waveform.astype('<f4').tobytes()).hexdigest(),
              'reference_sha256': sha256(args.reference), 'audio_contract': CONTRACT,
              'frames': frames, 'fps': 16, 'clip_seconds': frames/16,
              'source_audio_start_seconds': start, 'audio_alignment': alignment,
              'audio_duplicates_training_prefix': False, 'matched_teacher_used': False,
              'same_reference_pixels_as_training': bool(np.array_equal(normalized_reference, training['reference'])),
              'new_person_generalization_verified': False, 'lip_sync_verified': False,
              'generalization_verified': False, 'phone_validated': False, 'video_quality_validated': False,
              'silence_control_output_change_mae': float((prediction-silence_prediction).abs().mean()),
              'dark_area_std': float(scores.std()), 'dark_area_min': float(scores.min()),
              'dark_area_max': float(scores.max()), 'all_frames_finite': True,
              'limitations': ['A different audio file and moving lips do not prove phoneme-level sync.',
                              'No matched teacher or human sync acceptance; fixed mouth crop and original person only.',
                              'CPU PyTorch preview; no Android runtime or playback-latency conclusion.']}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--training-manifest', type=Path, required=True)
    parser.add_argument('--audio-source', type=Path, required=True, help='WAV, MP3 or a video containing audio')
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=10)
    args = parser.parse_args()
    if not .5 <= args.max_seconds <= 10:
        parser.error('--max-seconds must be .5..10')
    infer(args)


if __name__ == '__main__':
    main()
