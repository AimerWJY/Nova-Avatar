"""Fetch and verify the pinned official ncnn Android runtime (host-side only)."""
import argparse
import hashlib
import json
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / 'configs/ncnn-runtime.json'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def safe_member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name:
        raise ValueError(f'unsafe archive path: {name}')
    return path


def install(archive=None):
    config = json.loads(CONFIG.read_text())
    cache = REPO / 'cache/mobile'
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / config['filename']
    if archive is None and not destination.exists():
        if not shutil.which('curl'):
            raise ValueError('curl missing; use --archive with an already downloaded official ZIP')
        partial = destination.with_suffix('.zip.partial')
        command = ['curl', '--http1.1', '--fail', '--location', '--connect-timeout', '15',
                   '--max-time', '90', '--continue-at', '-', '--retry', '2', '--retry-delay', '2', '--retry-all-errors',
                   '--output', str(partial), config['url']]
        print(f"Downloading official ncnn {config['version']} (~29MB)...", flush=True)
        result = subprocess.run(command)
        if result.returncode:
            raise RuntimeError('download failed; partial retained; download official ZIP separately and pass --archive')
        if digest(partial) != config['sha256']:
            raise ValueError('download SHA256 mismatch; partial retained, not extracted')
        partial.replace(destination)
    source = archive.resolve(strict=True) if archive else destination
    if digest(source) != config['sha256']:
        raise ValueError('official runtime SHA256 mismatch; refusing to extract or execute')
    folder = cache / f"ncnn-{config['version']}"
    folder.mkdir(exist_ok=True)
    with zipfile.ZipFile(source) as package:
        for member in package.infolist():
            path = safe_member(member.filename)
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError('symlinks in runtime archive are not accepted')
            # Keep only this device ABI and licensing metadata.
            if config['abi'] not in path.parts and not any('license' in part.lower() for part in path.parts):
                continue
            package.extract(member, folder)
    headers = list(folder.rglob('net.h'))
    libraries = list(folder.rglob('libncnn.so'))
    if len(headers) != 1 or len(libraries) != 1:
        raise ValueError('unexpected official archive layout; inspect cache before proceeding')
    record = {'version': config['version'], 'archive_sha256': config['sha256'],
              'archive': str(source), 'include': str(headers[0].parent),
              'library': str(libraries[0]), 'library_dir': str(libraries[0].parent),
              'source_url': config['url']}
    record['library_sha256'] = digest(libraries[0])
    (folder / 'runtime.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='official fixed-version ZIP downloaded separately')
    args = parser.parse_args()
    try:
        install(args.archive)
        return 0
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
