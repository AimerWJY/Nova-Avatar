"""Read an authorized Android device profile via ADB; no APK installation."""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROPERTIES = ('ro.product.model', 'ro.product.manufacturer', 'ro.product.device',
              'ro.soc.model', 'ro.soc.manufacturer', 'ro.product.cpu.abilist',
              'ro.build.version.release', 'ro.build.version.sdk',
              'ro.build.version.security_patch', 'dalvik.vm.heapgrowthlimit',
              'dalvik.vm.heapsize')


def execute(command, timeout=20):
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or 'ADB command failed')
    return result.stdout.strip()


def devices(text):
    result = []
    for line in text.splitlines():
        columns = line.split()
        if len(columns) >= 2 and columns[0] != 'List' and not line.startswith('*'):
            result.append({'serial': columns[0], 'state': columns[1]})
    return result


def choose_device(items, serial=None):
    if serial:
        selected = next((item for item in items if item['serial'] == serial), None)
        if selected is None or selected['state'] != 'device':
            raise ValueError('selected device is missing, offline or unauthorized')
        return serial
    ready = [item['serial'] for item in items if item['state'] == 'device']
    if len(ready) != 1:
        raise ValueError('connect and authorize one device, or select it with --serial')
    return ready[0]


def memory_info(text):
    values = {}
    for name in ('MemTotal', 'MemAvailable', 'MemFree'):
        match = re.search(rf'^{name}:\s+(\d+)\s+kB\s*$', text, re.MULTILINE)
        if match:
            values[name] = int(match.group(1)) * 1024
    if 'MemTotal' not in values or 'MemAvailable' not in values:
        raise ValueError('missing MemTotal or MemAvailable in /proc/meminfo')
    return values


def sdk_tools():
    roots = [os.environ.get('ANDROID_HOME'), os.environ.get('ANDROID_SDK_ROOT'),
             str(Path.home() / 'Library/Android/sdk')]
    root = next((Path(path) for path in roots if path and Path(path).is_dir()), None)
    tools = {'adb': shutil.which('adb'), 'java': shutil.which('java'),
             'sdk': str(root) if root else None}
    for name in ('ndk', 'cmake', 'platforms', 'build-tools'):
        folder = root / name if root else None
        tools[name] = sorted(p.name for p in folder.iterdir() if p.is_dir()) if folder and folder.is_dir() else []
    return tools


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial')
    parser.add_argument('--output', type=Path, default=Path('reports/mobile'))
    args = parser.parse_args()
    try:
        host = sdk_tools()
        adb = host['adb']
        if not adb:
            raise ValueError('adb not on PATH; install Android SDK Platform Tools')
        serial = choose_device(devices(execute([adb, 'devices'])), args.serial)
        base = [adb, '-s', serial, 'shell']
        properties = {name: execute(base + ['getprop', name]) for name in PROPERTIES}
        memory = memory_info(execute(base + ['cat', '/proc/meminfo']))
        report = {'schema_version': 1, 'serial': serial,
                  'captured_at_utc': datetime.now(timezone.utc).isoformat(),
                  'properties': properties, 'memory_bytes': memory, 'host_tools': host,
                  'notes': ['MemAvailable is a system snapshot, not the app allocation budget.',
                            'Dalvik heap limits do not define the native/GPU allocation limit.',
                            'This profile is not an inference or NPU compatibility benchmark.']}
        args.output.mkdir(parents=True, exist_ok=True)
        destination = args.output / f'profile_{uuid.uuid4().hex[:12]}.json'
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f'Report: {destination.resolve()}')
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
