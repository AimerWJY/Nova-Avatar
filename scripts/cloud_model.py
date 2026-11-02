"""Inspect shards or assemble a local model view without changing shared storage."""
import argparse
import json
import struct
import sys
from pathlib import Path

INDEX = 'diffusion_pytorch_model.safetensors.index.json'
REPO = 'Wan-AI/Wan2.2-S2V-14B'
COMPONENTS = ('config.json', 'Wan2.1_VAE.pth', 'models_t5_umt5-xxl-enc-bf16.pth',
              'google/umt5-xxl', 'wav2vec2-large-xlsr-53-english')


def weight_map(root):
    mapping = json.loads((root / INDEX).read_text())['weight_map']
    if not mapping:
        raise ValueError('empty weight map')
    for name in set(mapping.values()):
        if Path(name).name != name or name in ('.', '..') or '\\' in name:
            raise ValueError(f'unsafe shard name: {name}')
    return mapping


def check_header(path, expected_keys):
    # Read tensor metadata only, not tens of GB of tensor payloads.
    with path.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f'truncated header: {path}')
        length = struct.unpack('<Q', prefix)[0]
        if length > 100 * 1024 * 1024 or length + 8 > path.stat().st_size:
            raise ValueError(f'invalid header length: {path}')
        metadata = json.loads(stream.read(length))
    keys = set(metadata) - {'__metadata__'}
    if keys != set(expected_keys):
        raise ValueError(f'index/header keys differ: {path}')
    payload = path.stat().st_size - 8 - length
    for key in keys:
        start, end = metadata[key]['data_offsets']
        if not 0 <= start <= end <= payload:
            raise ValueError(f'truncated/invalid tensor payload: {path}: {key}')


def inspect(root):
    mapping = weight_map(root)
    missing = []
    for name in sorted(set(mapping.values())):
        path = root / name
        if not path.is_file():
            missing.append(name)
            print(f'MISSING: {name}')
        else:
            check_header(path, [key for key, shard in mapping.items() if shard == name])
            print(f'OK: {name} ({path.stat().st_size:,} bytes; header checked)')
    for name in COMPONENTS:
        exists = (root / name).exists()
        print(f'{"OK" if exists else "MISSING"}: {name}')
        if not exists:
            missing.append(name)
    return missing


def repair(source, target):
    source = source.resolve(strict=True)
    if target.is_symlink():
        raise ValueError('target must not be a symlink')
    target = target.resolve()
    if target == source or source in target.parents or target in source.parents:
        raise ValueError('source and target must be separate non-nested directories')
    mapping = weight_map(source)
    for name in COMPONENTS:
        if not (source / name).exists():
            raise ValueError(f'non-shard component missing: {name}; repair only downloads shards')
    shards = set(mapping.values())
    missing = sorted(name for name in shards if not (source / name).is_file())
    # Validate existing source headers before trusting the local index.
    inspect(source)
    target.mkdir(parents=True, exist_ok=True)
    # Refuse conflicting entries before starting a large download.
    for item in source.iterdir():
        if not item.exists() or item.name in missing:
            continue
        destination = target / item.name
        if destination.exists() or destination.is_symlink():
            if not destination.is_symlink() or destination.resolve() != item.resolve():
                raise ValueError(f'conflicting local entry; left untouched: {destination}')
    for name in missing:
        destination = target / name
        if destination.is_symlink():
            raise ValueError(f'local shard must not be a symlink: {destination}')
        if destination.exists():
            check_header(destination, [key for key, shard in mapping.items() if shard == name])
    needed = [name for name in missing if not (target / name).exists()]
    revision = None
    if needed:
        from huggingface_hub import HfApi, hf_hub_download
        revision = HfApi().model_info(REPO).sha
        remote_index = hf_hub_download(REPO, INDEX, revision=revision)
        remote_mapping = json.loads(Path(remote_index).read_text())['weight_map']
        if mapping != remote_mapping:
            raise ValueError('remote/local index mismatch; refusing to mix checkpoints')
        for name in needed:
            print(f'Downloading only {name}; revision={revision}', flush=True)
            hf_hub_download(REPO, name, revision=revision, local_dir=str(target))
    for item in source.iterdir():
        if not item.exists() or item.name in missing:
            continue
        destination = target / item.name
        if not destination.exists() and not destination.is_symlink():
            destination.symlink_to(item, target_is_directory=item.is_dir())
    if inspect(target):
        raise ValueError('assembled model is still incomplete')
    if revision:
        (target / 'nova-download-record.json').write_text(json.dumps({
            'repo': REPO, 'revision': revision, 'downloaded_shards': needed,
            'shared_source': str(source), 'validation': 'index, header keys and payload bounds; not full checksum'
        }, indent=2) + '\n')
    print(f'Model view ready: {target}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'repair'])
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == 'repair':
            repair(args.source, args.target)
            return 0
        root = args.target if (args.target / INDEX).is_file() else args.source
        print(f'Checking: {root}')
        return 1 if inspect(root) else 0
    except (OSError, ValueError, KeyError, ImportError, struct.error) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
