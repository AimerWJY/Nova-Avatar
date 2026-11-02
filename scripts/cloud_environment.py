"""Collect cloud prerequisites without downloading or loading model weights."""
import argparse
import importlib
import json
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'python': sys.version, 'executable': sys.executable,
              'packages': {}, 'tools': {}, 'errors': []}
    for name in ('torch', 'torchvision', 'transformers', 'diffusers', 'accelerate',
                 'safetensors', 'numpy', 'cv2', 'librosa', 'soundfile', 'decord',
                 'einops', 'easydict', 'ftfy', 'imageio', 'sentencepiece'):
        try:
            package = importlib.import_module(name)
            report['packages'][name] = getattr(package, '__version__', 'import OK')
            if name == 'torch':
                available = package.cuda.is_available()
                report['cuda'] = {'available': available, 'runtime': package.version.cuda}
                if available:
                    report['cuda']['devices'] = [
                        {'name': package.cuda.get_device_name(i),
                         'vram_bytes': package.cuda.get_device_properties(i).total_memory}
                        for i in range(package.cuda.device_count())]
                else:
                    report['errors'].append('CUDA unavailable')
        except Exception as exc:
            report['errors'].append(f'{name}: {type(exc).__name__}: {exc}')
    for tool in ('git', 'ffmpeg', 'ffprobe', 'nvidia-smi'):
        report['tools'][tool] = shutil.which(tool)
        if not report['tools'][tool]:
            report['errors'].append(f'{tool} missing')
    usage = shutil.disk_usage(args.root)
    report['disk'] = {'root': str(args.root), 'total_bytes': usage.total, 'free_bytes': usage.free}
    result = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
    report['pip_check'] = result.stdout + result.stderr
    if result.returncode:
        report['errors'].append('pip check failed')
    args.output.mkdir(parents=True, exist_ok=True)
    token = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    destination = args.output / f'environment_{token}.json'
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f'Report: {destination}')
    return 1 if report['errors'] else 0


if __name__ == '__main__':
    sys.exit(main())
