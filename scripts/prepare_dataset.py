#!/usr/bin/env python3
"""Inspect or ingest paired audiovisual sources; Python standard library only."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

VIDEO_EXTENSIONS = {'.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v'}


def run(args):
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode:
        raise ValueError(result.stderr.strip()[-2000:] or 'command failed')
    return result.stdout


def probe(path):
    return json.loads(run(['ffprobe', '-v', 'error', '-show_streams',
                           '-show_format', '-of', 'json', str(path)]))


def number(value):
    try:
        value = float(Fraction(str(value)))
        return value if value > 0 else None
    except (ValueError, ZeroDivisionError, TypeError):
        return None


def inspect(path):
    info = probe(path)
    video = next((s for s in info['streams'] if s['codec_type'] == 'video'), None)
    audio = next((s for s in info['streams'] if s['codec_type'] == 'audio'), None)
    if not video or not audio:
        raise ValueError('requires both a video stream and an embedded audio stream')
    duration = number(video.get('duration')) or number(info.get('format', {}).get('duration'))
    if not duration:
        raise ValueError('unable to determine duration')
    warnings = []
    audio_duration = number(audio.get('duration'))
    if audio_duration is None:
        warnings.append('audio duration unknown; manually verify alignment')
    elif abs(duration - audio_duration) > 0.1:
        warnings.append('audio/video duration difference exceeds 100ms')
    if duration < 3:
        warnings.append('shorter than 3 seconds')
    if min(video['width'], video['height']) < 480:
        warnings.append('short side below 480 pixels')
    vfr = None
    try:
        lines = run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                     '-show_entries', 'frame=best_effort_timestamp_time',
                     '-of', 'csv=p=0', str(path)]).splitlines()
        times = []
        for line in lines:
            try:
                times.append(float(line.split(',')[0]))
            except ValueError:
                continue
        deltas = [b - a for a, b in zip(times, times[1:])]
        if len(deltas) < 2:
            warnings.append('insufficient frame timestamps for VFR check')
        else:
            vfr = max(deltas) - min(deltas) > 0.001
            if vfr:
                warnings.append('variable/nonuniform frame timing detected')
    except ValueError:
        warnings.append('frame timestamp inspection failed')
    return {'duration_seconds': duration, 'width': video['width'],
            'height': video['height'], 'fps': number(video.get('avg_frame_rate')),
            'num_frames': number(video.get('nb_frames')), 'variable_frame_timing': vfr,
            'source_audio_sample_rate': number(audio.get('sample_rate')),
            'source_audio_duration_seconds': audio_duration}, warnings


def content_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_asset(target, producer):
    if target.exists():
        probe(target)  # Refuse to silently reuse a corrupt partial file.
        return
    temporary = target.with_name(f'.{target.stem}.{uuid.uuid4().hex}{target.suffix}')
    try:
        producer(temporary)
        probe(temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['scan', 'prepare'])
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--person-id', required=True)
    parser.add_argument('--source-type', choices=['real', 'synthetic'], required=True)
    parser.add_argument('--rights-note', required=True)
    args = parser.parse_args()
    required = ['ffprobe'] + (['ffmpeg'] if args.mode == 'prepare' else [])
    for tool in required:
        if not shutil.which(tool):
            parser.error(f'{tool} not found on PATH; install FFmpeg first')
    source = args.input.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        parser.error('--input must be an existing directory')
    if output == source or source in output.parents or output in source.parents:
        parser.error('input and output must be separate, non-nested directories')
    files = sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
    if not files:
        parser.error('no supported video files found')
    manifests = output / 'manifests'
    manifests.mkdir(parents=True, exist_ok=True)
    if args.mode == 'prepare':
        for folder in ['originals', 'audios', 'references']:
            (output / folder).mkdir(exist_ok=True)
    token = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    records, errors, duplicates, seen = [], [], [], set()
    for path in files:
        try:
            sha = content_hash(path)
            sample_id = 'clip_' + hashlib.sha256(
                f'{sha}\0{args.person_id}\0{args.source_type}'.encode()).hexdigest()[:24]
            if sample_id in seen:
                duplicates.append(str(path))
                continue
            details, warnings = inspect(path)
            record = {'schema_version': 1, 'id': sample_id,
                      'source_id': sha, 'source_sha256': sha,
                      'source_filename': str(path.relative_to(source)),
                      'source_type': args.source_type, 'person_id': args.person_id,
                      'rights_note': args.rights_note, 'split': 'unassigned',
                      **details, 'caption': '', 'tags': {},
                      'review': {'status': 'pending', 'av_sync': 'pending',
                                 'face_visible': 'pending', 'hands_quality': 'pending', 'notes': ''},
                      'warnings': warnings}
            if args.source_type == 'synthetic':
                record['generation'] = {'model': None, 'revision': None, 'prompt': None,
                                        'seed': None, 'steps': None, 'sampler': None, 'cfg': None}
            if args.mode == 'prepare':
                original = output / 'originals' / (sample_id + path.suffix.lower())
                audio = output / 'audios' / (sample_id + '.wav')
                reference = output / 'references' / (sample_id + '.png')
                if original.exists() and content_hash(original) != sha:
                    raise ValueError('existing original content differs; refusing overwrite')
                atomic_asset(original, lambda dest: shutil.copyfile(path, dest))
                atomic_asset(audio, lambda dest: run([
                    'ffmpeg', '-nostdin', '-v', 'error', '-n', '-i', str(original),
                    '-map', '0:a:0', '-vn', '-ac', '1', '-ar', '16000',
                    '-c:a', 'pcm_s16le', str(dest)]))
                atomic_asset(reference, lambda dest: run([
                    'ffmpeg', '-nostdin', '-v', 'error', '-n', '-ss',
                    str(details['duration_seconds'] / 2), '-i', str(original),
                    '-map', '0:v:0', '-frames:v', '1', str(dest)]))
                record.update(video=str(original.relative_to(output)),
                              audio=str(audio.relative_to(output)),
                              reference_image=str(reference.relative_to(output)),
                              audio_sample_rate=16000)
            records.append(record)
            seen.add(sample_id)
        except (ValueError, OSError, KeyError) as exc:
            errors.append({'file': str(path), 'error': str(exc)})
    manifest = manifests / f'{args.mode}_{token}.jsonl'
    with manifest.open('x', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    report = {'mode': args.mode, 'input_files': len(files), 'accepted': len(records),
              'duplicates': duplicates, 'errors': errors,
              'pending_review': len(records), 'manifest': str(manifest)}
    report_path = manifests / f'report_{token}.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'accepted': len(records), 'errors': len(errors),
                      'manifest': str(manifest), 'report': str(report_path)}, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
