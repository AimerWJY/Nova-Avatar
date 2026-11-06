"""Prepare one explicitly reviewed teacher pair using its embedded audio timestamps."""
import hashlib
import json
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .audio import CONTRACT, features


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def square_image(image):
    image = ImageOps.exif_transpose(image).convert('RGB')
    return np.array(ImageOps.pad(image, (256, 256), method=Image.Resampling.LANCZOS,
                                color=(0, 0, 0), centering=(.5, .5)), dtype=np.uint8)


def audio_timeline(path, video_start, sample_count):
    import av
    resampler = av.AudioResampler(format='fltp', layout='mono', rate=16000)
    segments = []

    def append(converted):
        for frame in converted:
            if frame.pts is None or frame.time_base is None:
                raise ValueError('audio frame timestamps missing; refusing guessed alignment')
            start = round((float(frame.pts*frame.time_base)-video_start)*16000)
            segments.append((start, frame.to_ndarray().reshape(-1).astype(np.float32)))

    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError('teacher video must contain the matching audio track')
        for frame in container.decode(audio=0):
            append(resampler.resample(frame))
        append(resampler.resample(None))
    waveform = np.zeros(sample_count, dtype=np.float32)
    covered = np.zeros(sample_count, dtype=bool)
    for start, samples in segments:
        left, right = max(0, start), min(sample_count, start+len(samples))
        if right <= left:
            continue
        if covered[left:right].any():
            raise ValueError('overlapping decoded audio timestamps')
        waveform[left:right] = samples[left-start:right-start]
        covered[left:right] = True
    coverage = float(covered.mean())
    if coverage < .98 or not np.isfinite(waveform).all():
        raise ValueError(f'audio coverage insufficient or invalid: {coverage:.4f}')
    missing = np.flatnonzero(~covered)
    if len(missing) and (missing[0] == 0 or covered[missing[0]:].any()):
        raise ValueError('leading/interior audio timestamp gap; refusing inferred alignment')
    return waveform, {'coverage_fraction': coverage,
                      'first_audio_offset_seconds': segments[0][0]/16000 if segments else None,
                      'contiguous_audio_samples': int(missing[0]) if len(missing) else sample_count,
                      'uncovered_samples_zero_padded': int((~covered).sum())}


def prepare(video, reference, output, person_id, review_note):
    import av
    video, reference = video.resolve(strict=True), reference.resolve(strict=True)
    decoded, pts = [], []
    with av.open(str(video)) as container:
        if not container.streams.video or not container.streams.audio:
            raise ValueError('teacher video must contain both video and audio')
        for frame in container.decode(video=0):
            if frame.pts is None or frame.time_base is None:
                raise ValueError('video frame timestamp missing')
            pts.append(float(frame.pts*frame.time_base))
            decoded.append(square_image(frame.to_image()))
            if len(decoded) > 160:
                raise ValueError('initial overfit experiment accepts at most 160 frames; trim an explicit paired clip first')
    if len(decoded) < 8 or not np.allclose(np.diff(pts), 1/16, atol=.001, rtol=0):
        raise ValueError('requires at least 8 uniformly timed 16FPS frames; no implicit resampling')
    times = np.asarray(pts, dtype=np.float64)-pts[0]
    duration = float(times[-1]+1/16)
    waveform, audio_info = audio_timeline(video, pts[0], round(duration*16000))
    original_frames = len(decoded)
    retained = min(original_frames, audio_info['contiguous_audio_samples']//1000)
    if retained < 8:
        raise ValueError('fewer than 8 full video intervals covered by decoded audio')
    decoded, times = decoded[:retained], times[:retained]
    duration = retained/16
    waveform = waveform[:retained*1000]
    with Image.open(reference) as image:
        ref = square_image(image)
    output.mkdir(parents=True, exist_ok=False)
    cache = output / 'sample.npz'
    np.savez_compressed(cache, reference=ref, frames=np.stack(decoded),
                        audio_features=features(waveform, times), audio=waveform, times=times)
    with wave.open(str(output / 'aligned_audio.wav'), 'wb') as stream:
        stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(16000)
        stream.writeframes(np.rint(np.clip(waveform, -1, 1)*32767).astype('<i2').tobytes())
    Image.fromarray(ref).save(output / 'reference.png')
    record = {'schema_version': 1, 'format': 'candidate_teacher_pair_v1', 'person_id': person_id,
              'source_type': 'synthetic', 'teacher_model': 'Wan2.2-S2V-14B',
              'generation_report': None, 'generation_metadata_status': 'not_supplied_for_this_clip',
              'source_video': str(video), 'source_reference': str(reference),
              'source_video_sha256': sha256(video), 'source_reference_sha256': sha256(reference),
              'cache_file': cache.name, 'cache_sha256': sha256(cache), 'frames': len(decoded),
              'fps': 16, 'duration_seconds': duration, 'video_start_pts_seconds': pts[0],
              'original_decoded_frames': original_frames,
              'tail_frames_removed_for_audio_coverage': original_frames-retained,
              'preprocessing': '256 square, aspect-preserving resize with centered black letterbox; no face crop',
              'audio_alignment': audio_info, 'audio_contract': CONTRACT,
              'review': {'status': 'approved', 'basis': review_note},
              'split': 'overfit_only', 'independent_validation': False}
    (output / 'manifest.json').write_text(json.dumps(record, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(record, indent=2, ensure_ascii=False))
    return record


def load(manifest):
    record = json.loads(manifest.read_text())
    if (record.get('schema_version') != 1 or record.get('format') != 'candidate_teacher_pair_v1'
            or record.get('review', {}).get('status') != 'approved'
            or record.get('split') != 'overfit_only' or record.get('audio_contract') != CONTRACT
            or record.get('cache_file') != 'sample.npz'):
        raise ValueError('unreviewed/incompatible overfit dataset manifest')
    cache = manifest.resolve().parent / 'sample.npz'
    if sha256(cache) != record['cache_sha256']:
        raise ValueError('teacher cache SHA256 mismatch')
    with np.load(cache, allow_pickle=False) as archive:
        data = {key: archive[key] for key in ('reference', 'frames', 'audio_features', 'audio', 'times')}
    count = record['frames']
    if (type(count) is not int or not 8 <= count <= 160 or record.get('fps') != 16
            or data['frames'].shape != (count, 256, 256, 3)
            or data['reference'].shape != (256, 256, 3) or data['audio_features'].shape != (count, 40)
            or data['audio'].shape != (count*1000,) or data['times'].shape != (count,)
            or data['frames'].dtype != np.uint8 or data['reference'].dtype != np.uint8
            or data['audio_features'].dtype != np.float32 or data['audio'].dtype != np.float32
            or not all(np.isfinite(value).all() for value in data.values())
            or not np.allclose(data['times'], np.arange(count)/16, atol=.001, rtol=0)):
        raise ValueError('invalid teacher cache shapes/types/timestamps')
    return record, data
