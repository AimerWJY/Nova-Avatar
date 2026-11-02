"""Check dependencies, validating the known decord 0.6.0 wheel-tag defect."""
import importlib.metadata
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

KNOWN_MESSAGE = 'decord 0.6.0 is not supported on this platform'
KNOWN_TAG = 'cp36-cp36m-manylinux2010_x86_64'
ISSUE = 'https://github.com/dmlc/decord/issues/356'


def decode_smoke_test():
    """Exercise CPU video decoding in a bounded subprocess, not just import."""
    with tempfile.TemporaryDirectory(prefix='nova-decord-') as folder:
        video = Path(folder) / 'test.mp4'
        subprocess.run([
            'ffmpeg', '-nostdin', '-v', 'error', '-f', 'lavfi', '-i',
            'color=c=blue:s=64x64:r=5', '-frames:v', '5', '-c:v', 'mpeg4',
            '-pix_fmt', 'yuv420p', str(video)
        ], check=True, capture_output=True, text=True, timeout=30)
        code = '''import json, sys
from decord import VideoReader, cpu
reader = VideoReader(sys.argv[1], ctx=cpu(0), num_threads=1)
assert len(reader) == 5, len(reader)
frames = reader.get_batch([0, 2, 4]).asnumpy()
assert frames.shape == (3, 64, 64, 3), frames.shape
assert frames.dtype.name == 'uint8', frames.dtype
assert frames[..., 2].mean() > 200, 'unexpected blue frame contents'
print(json.dumps({'frames': len(reader), 'batch_shape': list(frames.shape), 'decode': 'pass'}))
'''
        decoded = subprocess.run([sys.executable, '-c', code, str(video)],
                                 check=True, capture_output=True, text=True, timeout=30)
        return json.loads(decoded.stdout)


def classify(result, version, wheel, system, machine, smoke_test):
    raw = result.stdout + result.stderr
    report = {'raw_output': raw, 'returncode': result.returncode,
              'errors': [], 'warnings': []}
    if result.returncode == 0:
        return report
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    tags = [line.split(':', 1)[1].strip() for line in wheel.splitlines() if line.startswith('Tag:')]
    recognized = (lines == [KNOWN_MESSAGE] and version == '0.6.0' and
                  tags == [KNOWN_TAG] and system == 'Linux' and machine == 'x86_64')
    report['decord_metadata'] = {'version': version, 'wheel_tags': tags}
    if not recognized:
        report['errors'].append('pip check failed; not the narrowly recognized decord wheel-tag defect')
        return report
    try:
        report['decord_decode_test'] = smoke_test()
        report['warnings'].append(
            f'Known decord WHEEL metadata defect; CPU decode test passed. Raw pip check remains nonzero. {ISSUE}')
    except Exception as exc:
        details = getattr(exc, 'stderr', None) or str(exc)
        report['errors'].append(f'decord video decode test failed: {details}')
    return report


def check_dependencies():
    result = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
    version, wheel = None, ''
    if result.returncode:
        try:
            distribution = importlib.metadata.distribution('decord')
            version = distribution.version
            wheel = distribution.read_text('WHEEL') or ''
        except importlib.metadata.PackageNotFoundError:
            pass
    return classify(result, version, wheel, platform.system(), platform.machine(), decode_smoke_test)


if __name__ == '__main__':
    report = check_dependencies()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    sys.exit(1 if report['errors'] else 0)
