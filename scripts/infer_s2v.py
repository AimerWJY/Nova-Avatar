"""Offline S2V teacher baseline with local weights and DiffSynth VRAM management."""
import argparse
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from cloud_model import DEFAULT_SOURCE, inspect, weight_map

FPS = 16
SAMPLE_RATE = 16000
PIN_FILE = Path(__file__).resolve().parents[1] / 'configs/diffsynth-revision.txt'
PROMPT = ('A single person speaks naturally to the camera, with subtle facial expressions, '
          'gentle head and shoulder movements and occasional small natural hand gestures. '
          'Fixed camera, stable identity, simple static background.')
NEGATIVE = ('blurry, distorted face, deformed hands, extra fingers, fused fingers, '
            'extra limbs, duplicate person, subtitles, watermark, camera movement')


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'{command[0]} failed: {result.stderr.strip()[-3000:]}')
    return result.stdout


def probe_audio(path):
    info = json.loads(run(['ffprobe', '-v', 'error', '-show_streams', '-show_format',
                           '-of', 'json', str(path)]))
    streams = [s for s in info['streams'] if s['codec_type'] == 'audio']
    if not streams:
        raise ValueError(f'no audio stream: {path}')
    stream = streams[0]
    duration = float(stream.get('duration') or info['format']['duration'])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('audio duration must be finite and positive')
    return {'duration_seconds': duration, 'sample_rate': int(stream['sample_rate']),
            'channels': stream['channels'], 'codec': stream['codec_name']}


def frame_count(duration, seconds):
    # S2V drops its reference frame. Output frames must be divisible by four.
    return math.floor(min(duration, seconds) * FPS / 4 + 1e-8) * 4 + 1


def asset_paths(root):
    return {
        'dit': [str(root / name) for name in sorted(set(weight_map(root).values()))],
        'audio_encoder': str(root / 'wav2vec2-large-xlsr-53-english/model.safetensors'),
        'text_encoder': str(root / 'models_t5_umt5-xxl-enc-bf16.pth'),
        'vae': str(root / 'Wan2.1_VAE.pth'),
        'tokenizer': str(root / 'google/umt5-xxl'),
        'audio_processor': str(root / 'wav2vec2-large-xlsr-53-english'),
    }


def validate_settings(args):
    if args.width <= 0 or args.height <= 0 or args.width % 16 or args.height % 16:
        raise ValueError('width/height must be positive multiples of 16')
    if not math.isfinite(args.seconds) or not 1 <= args.seconds <= 5:
        raise ValueError('this single-clip baseline supports --seconds from 1 to 5')
    if args.steps < 1:
        raise ValueError('steps must be positive')
    if not math.isfinite(args.vram_limit) or args.vram_limit <= 0:
        raise ValueError('vram-limit must be positive GiB')


def make_plan(args):
    validate_settings(args)
    for tool in ('ffmpeg', 'ffprobe'):
        if not shutil.which(tool):
            raise ValueError(f'{tool} is required')
    root = args.model_dir.resolve(strict=True)
    if inspect(root):
        raise ValueError('model is incomplete; model-check must pass first')
    assets = asset_paths(root)
    for paths in assets.values():
        for path in paths if isinstance(paths, list) else [paths]:
            if not Path(path).exists():
                raise ValueError(f'missing local model asset: {path}')
    image = (args.image or args.root / 'inputs/person_01.png').resolve(strict=True)
    audio = (args.audio or args.root / 'inputs/normal.wav').resolve(strict=True)
    from PIL import Image
    with Image.open(image) as source:
        source.verify()
    with Image.open(image) as source:
        image_size = list(source.size)
    audio_info = probe_audio(audio)
    if audio_info['duration_seconds'] < 1:
        raise ValueError('at least one second of audio is required')
    frames = frame_count(audio_info['duration_seconds'], args.seconds)
    return {'model_dir': str(root), 'assets': assets, 'image': str(image),
            'audio': str(audio), 'original_image_size': image_size,
            'source_audio': audio_info, 'width': args.width, 'height': args.height,
            'fps': FPS, 'num_frames': frames, 'output_frames': frames - 1,
            'clip_seconds': (frames - 1) / FPS, 'steps': args.steps,
            'seed': args.seed, 'vram_limit_gib': args.vram_limit, 'offload': args.offload,
            'prompt': args.prompt, 'negative_prompt': args.negative_prompt,
            'image_resize': 'aspect-preserving letterbox; no body crop',
            'expected_diffsynth_revision': PIN_FILE.read_text().strip(),
            'mode': 'offline teacher baseline; not streaming'}


def build_pipeline(plan, torch, pipeline_class, config_class):
    dtype = torch.bfloat16
    disk = plan['offload'] == 'disk'
    memory = dict(offload_device='disk' if disk else 'cpu',
                  offload_dtype='disk' if disk else dtype,
                  onload_device='cpu', onload_dtype=dtype,
                  preparing_device='cuda', preparing_dtype=dtype,
                  computation_device='cuda', computation_dtype=dtype)
    assets = plan['assets']
    configs = [config_class(path=assets[name], **memory)
               for name in ('dit', 'audio_encoder', 'text_encoder', 'vae')]
    return pipeline_class.from_pretrained(
        torch_dtype=dtype, device='cuda', model_configs=configs,
        tokenizer_config=config_class(path=assets['tokenizer']),
        audio_processor_config=config_class(path=assets['audio_processor']),
        redirect_common_files=False, vram_limit=plan['effective_vram_limit_gib'])


def backend_provenance():
    distribution = importlib.metadata.distribution('diffsynth')
    direct_url = json.loads(distribution.read_text('direct_url.json') or '{}')
    commit = direct_url.get('vcs_info', {}).get('commit_id')
    expected = PIN_FILE.read_text().strip()
    expected_archive = f'https://codeload.github.com/modelscope/DiffSynth-Studio/zip/{expected}'
    archive_matches = (direct_url.get('url') == expected_archive and
                       isinstance(direct_url.get('archive_info'), dict))
    if commit != expected and not archive_matches:
        raise ValueError('DiffSynth revision differs from the project pin; run install-diffsynth')
    return {'version': distribution.version, 'revision': expected,
            'installation': 'commit_archive' if archive_matches else 'git',
            'source_url': direct_url.get('url'),
            'archive_hashes': direct_url.get('archive_info', {}).get('hashes', {})}


def execute(plan, directory, report):
    import torch
    if not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; stop before loading weights')
    report['backend'] = backend_provenance()
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
    from PIL import Image, ImageOps
    import imageio.v2 as imageio
    import soundfile as sf
    os.environ['DIFFSYNTH_SKIP_DOWNLOAD'] = 'True'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    limit = min(plan['vram_limit_gib'], total_bytes / 1024**3 - 3, free_bytes / 1024**3 - 3)
    if limit < 4:
        raise ValueError('insufficient free VRAM; stop other GPU jobs')
    plan['effective_vram_limit_gib'] = limit
    report['gpu'] = {'name': torch.cuda.get_device_name(0), 'total_bytes': total_bytes,
                     'free_bytes_before_load': free_bytes, 'torch': torch.__version__,
                     'cuda_runtime': torch.version.cuda}
    prepared_audio = directory / 'audio.wav'
    run(['ffmpeg', '-nostdin', '-v', 'error', '-n', '-i', plan['audio'],
         '-map', '0:a:0', '-t', str(plan['clip_seconds']), '-vn',
         '-ac', '1', '-ar', str(SAMPLE_RATE), '-c:a', 'pcm_s16le', str(prepared_audio)])
    audio, rate = sf.read(prepared_audio, dtype='float32')
    if rate != SAMPLE_RATE or audio.ndim != 1:
        raise ValueError('unexpected preprocessed audio format')
    with Image.open(plan['image']) as source:
        image = ImageOps.pad(ImageOps.exif_transpose(source).convert('RGB'),
                             (plan['width'], plan['height']),
                             method=Image.Resampling.LANCZOS, color=(224, 224, 224))
    image.save(directory / 'reference.png')
    timings = report.setdefault('timings_seconds', {})
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    print('Loading local weights with VRAM management...', flush=True)
    pipe = build_pipeline(plan, torch, WanVideoPipeline, ModelConfig)
    torch.cuda.synchronize()
    timings['load'] = time.perf_counter() - started
    report['load_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    print(f"Generating {plan['clip_seconds']} seconds, {plan['steps']} steps...", flush=True)
    with torch.inference_mode():
        video = pipe(prompt=plan['prompt'], negative_prompt=plan['negative_prompt'],
                     input_image=image, input_audio=audio, audio_sample_rate=SAMPLE_RATE,
                     height=plan['height'], width=plan['width'], num_frames=plan['num_frames'],
                     num_inference_steps=plan['steps'], seed=plan['seed'],
                     cfg_scale=4.5, sigma_shift=3.0, tiled=True)
    torch.cuda.synchronize()
    timings['generate_including_conditioning_and_vae'] = time.perf_counter() - started
    report['generation_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    report['generation_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    if len(video) != plan['num_frames']:
        raise ValueError(f"unexpected output length: {len(video)}; expected {plan['num_frames']}")
    frames = video[1:]  # Match upstream S2V example: omit the reference frame.
    started = time.perf_counter()
    silent = directory / 'video_silent.mp4'
    with imageio.get_writer(str(silent), fps=FPS, codec='libx264', quality=7) as writer:
        import numpy as np
        for frame in frames:
            writer.append_data(np.asarray(frame))
    output = directory / 'video.mp4'
    run(['ffmpeg', '-nostdin', '-v', 'error', '-n', '-i', str(silent),
         '-i', str(prepared_audio), '-map', '0:v:0', '-map', '1:a:0',
         '-c:v', 'copy', '-c:a', 'aac', '-shortest', '-movflags', '+faststart', str(output)])
    info = json.loads(run(['ffprobe', '-v', 'error', '-show_streams', '-show_format',
                           '-of', 'json', str(output)]))
    if not {'video', 'audio'} <= {s['codec_type'] for s in info['streams']}:
        raise ValueError('output lacks audio or video')
    timings['encode_and_mux'] = time.perf_counter() - started
    report['output'] = str(output)
    report['output_duration_seconds'] = float(info['format']['duration'])
    report['generation_fps'] = len(frames) / timings['generate_including_conditioning_and_vae']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/cloud/nova-avatar'))
    parser.add_argument('--model-dir', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--audio', type=Path)
    parser.add_argument('--width', type=int, default=480)
    parser.add_argument('--height', type=int, default=640)
    parser.add_argument('--seconds', type=float, default=5)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--vram-limit', type=float, default=20)
    parser.add_argument('--offload', choices=['disk', 'cpu'], default='disk')
    parser.add_argument('--prompt', default=PROMPT)
    parser.add_argument('--negative-prompt', default=NEGATIVE)
    parser.add_argument('--dry-run', action='store_true', help='validate inputs/models; do not import torch or load weights')
    args = parser.parse_args()
    try:
        plan = make_plan(args)
    except Exception as exc:
        print(f'Preflight failed: {exc}', file=sys.stderr)
        return 1
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    token = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    directory = args.root.resolve() / 'outputs' / f's2v_{token}'
    # Never write generated artifacts into the shared model tree.
    model_root = args.model_dir.resolve()
    if directory == model_root or model_root in directory.parents:
        print('Output directory must be outside the model directory', file=sys.stderr)
        return 1
    directory.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'plan': plan, 'started_at_utc': token}
    report_path = directory / 'report.json'
    started = time.perf_counter()
    try:
        execute(plan, directory, report)
        report['status'] = 'success'
        return 0
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
        return 1
    finally:
        report.setdefault('timings_seconds', {})['total_after_preflight'] = time.perf_counter() - started
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Report: {report_path}', flush=True)
        if report.get('output'):
            print(f"Video: {report['output']}", flush=True)


if __name__ == '__main__':
    sys.exit(main())
