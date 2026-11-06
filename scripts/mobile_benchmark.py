"""Build and run a bounded native Android operator/driver probe over ADB."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from mobile_profile import choose_device, devices, execute, sdk_tools

REPO = Path(__file__).resolve().parents[1]


def build(output, ndk=None, backend='nnapi'):
    if backend not in ('nnapi', 'vulkan', 'ncnn', 'candidate', 'converted', 'preview'):
        raise ValueError('unknown backend')
    tools = sdk_tools()
    if ndk is None:
        if not tools['sdk'] or not tools['ndk']:
            raise ValueError('Android NDK missing; install it or pass --ndk')
        versions = sorted(tools['ndk'], key=lambda text: tuple(int(x) for x in text.split('.')))
        ndk = Path(tools['sdk']) / 'ndk' / versions[-1]
    hosts = sorted((ndk / 'toolchains/llvm/prebuilt').glob('*'))
    host_prefix = 'darwin-' if sys.platform == 'darwin' else 'linux-'
    host = next((path for path in hosts if path.name.startswith(host_prefix)), None)
    if host is None:
        raise ValueError('supported NDK toolchain not found for host')
    compiler = host / 'bin/aarch64-linux-android29-clang++'
    if not compiler.is_file():
        raise ValueError(f'compiler missing: {compiler}')
    output.mkdir(parents=True, exist_ok=True)
    binary = output / f'nova_{backend}_probe'
    source = REPO / f'mobile/native/{backend}_probe.cpp'
    if backend == 'preview':
        binary = output / 'libnova_avatar.so'
        source = REPO / 'mobile/android/avatar_jni.cpp'
    runtime = None
    if backend in ('ncnn', 'candidate', 'converted', 'preview'):
        from mobile_runtime import CONFIG, digest
        config = json.loads(CONFIG.read_text())
        record = REPO / 'cache/mobile' / f"ncnn-{config['version']}" / 'runtime.json'
        if not record.is_file():
            raise ValueError('ncnn runtime missing; run python3 scripts/mobile_runtime.py first')
        runtime = json.loads(record.read_text())
        if runtime['archive_sha256'] != config['sha256'] or digest(Path(runtime['archive'])) != config['sha256']:
            raise ValueError('ncnn archive provenance check failed')
        if digest(Path(runtime['library'])) != runtime['library_sha256']:
            raise ValueError('cached ncnn library changed; reinstall the verified archive')
    command = [str(compiler), '-std=c++17', '-O3', '-Wall', '-Wextra',
               '-Wno-deprecated-declarations', '-Wno-missing-field-initializers']
    command += ['-fPIC', '-shared'] if backend == 'preview' else ['-fPIE', '-pie']
    command.append(str(source))
    if runtime:
        command += ['-I' + runtime['include'], '-L' + runtime['library_dir'], '-lncnn', '-lvulkan',
                    '-Wl,-rpath,$ORIGIN']
        if backend == 'preview':
            command += ['-landroid']
    else:
        command += ['-static-libstdc++', '-lvulkan' if backend == 'vulkan' else '-lneuralnetworks']
    command += ['-o', str(binary)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    (output / 'build.log').write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f'NDK build failed: {result.stderr}')
    details = {'command': command, 'ndk': str(ndk), 'backend': backend, 'artifacts': [],
               'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest()}
    if runtime and backend != 'preview':
        common = REPO / 'mobile/native/ncnn_probe_common.hpp'
        details['common_header_sha256'] = hashlib.sha256(common.read_bytes()).hexdigest()
        if backend == 'candidate':
            fixture = REPO / 'mobile/native/candidate_fixture.hpp'
            details['fixture_header_sha256'] = hashlib.sha256(fixture.read_bytes()).hexdigest()
    if runtime:
        libraries = sorted(Path(runtime['library_dir']).glob('*.so'))
        if not any(path.name == 'libc++_shared.so' for path in libraries):
            libraries.append(host / 'sysroot/usr/lib/aarch64-linux-android/libc++_shared.so')
        details.update(runtime=runtime, artifacts=[str(path) for path in libraries],
                       artifact_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in libraries})
    if backend == 'vulkan':
        shader_source = REPO / 'mobile/native/matmul.comp'
        shader_output = output / 'matmul.spv'
        shader_tools = ndk / 'shader-tools' / host.name
        shader_command = [str(shader_tools / 'glslc'), '--target-env=vulkan1.1',
                          '-O', str(shader_source), '-o', str(shader_output)]
        result = subprocess.run(shader_command, capture_output=True, text=True, timeout=120)
        (output / 'shader.log').write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f'shader compilation failed: {result.stderr}')
        validation = subprocess.run([str(shader_tools / 'spirv-val'), '--target-env', 'vulkan1.1',
                                     str(shader_output)], capture_output=True, text=True, timeout=30)
        if validation.returncode:
            raise RuntimeError(f'SPIR-V validation failed: {validation.stderr}')
        details.update(shader_command=shader_command,
                       shader_source_sha256=hashlib.sha256(shader_source.read_bytes()).hexdigest(),
                       shader_binary_sha256=hashlib.sha256(shader_output.read_bytes()).hexdigest(),
                       artifacts=[str(shader_output)])
    return binary, details


def summarize(native):
    hardware, cpu, failed, unsupported = [], [], [], []
    for device in native['devices']:
        for result in device['results']:
            item = {'device': device['name'], **result}
            if result['status'] == 'success':
                (hardware if device['hardware_driver'] else cpu).append(item)
            elif result['status'] == 'unsupported':
                unsupported.append(item)
            else:
                failed.append(item)
    status = ('hardware_microbench_partial' if failed or unsupported else 'hardware_microbench_complete') if hardware else 'no_valid_hardware_results'
    return {'status': status,
            'hardware_results': hardware, 'cpu_results': cpu,
            'failed_results': failed, 'unsupported_results': unsupported,
            'can_infer_video_generation_speed': False}


def summarize_candidate(native):
    result = summarize(native)
    valid = (len(result['hardware_results']) == 1 and not result['failed_results']
             and not result['unsupported_results'] and type(native.get('trained')) is bool
             and native.get('synthetic_inputs') is True)
    sample = result['hardware_results'][0] if valid else {}
    valid = (valid and sample.get('all_frames_finite') is True
             and sample.get('cpu_reference_frames', 0) >= 8
             and sample.get('max_scaled_error', float('inf')) <= 1
             and sample.get('state_reset_max_scaled_error', float('inf')) <= 1)
    external = native.get('external_pytorch_reference', {})
    if external.get('available'):
        valid = valid and external.get('passed') is True
    result['status'] = 'candidate_graph_validated' if valid else 'candidate_graph_failed'
    frame_budget = 1000 / sample.get('target_fps', 16)
    result['synthetic_graph_warm_budget_met'] = bool(valid and sample['p95_ms'] <= frame_budget
        and sample['stream_wall_ms'] <= sample['frames'] * frame_budget)
    result['synthetic_graph_cold_2s_budget_met'] = bool(valid and sample['cold_first_rgb_ms'] <= 2000)
    result['video_quality_validated'] = False
    result['pytorch_equivalence_validated'] = bool(valid and external.get('available') and external.get('passed'))
    return result


def load_candidate_export(manifest):
    record = json.loads(manifest.read_text())
    if (not isinstance(record, dict) or record.get('schema_version') != 1
            or record.get('format') != 'nova_candidate_fixture_v1'
            or record.get('architecture') != 'recurrent_conditioned_rgb_v0'
            or record.get('bundle_file') != 'candidate.bundle'
            or record.get('reference_frames') != 8
            or type(record.get('frames')) is not int
            or not 8 <= record.get('frames', 0) <= 160):
        raise ValueError('unsupported candidate export manifest')
    bundle = manifest.resolve().parent / 'candidate.bundle'
    if bundle.is_symlink() or not bundle.is_file() or bundle.stat().st_size > 32*1024*1024:
        raise ValueError('missing, symlinked or oversized candidate bundle')
    if (bundle.stat().st_size != record.get('bundle_bytes')
            or hashlib.sha256(bundle.read_bytes()).hexdigest() != record.get('bundle_sha256')):
        raise ValueError('candidate bundle SHA256/size mismatch')
    return record, bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial')
    parser.add_argument('--backend', choices=['nnapi', 'vulkan', 'ncnn', 'candidate'], default='nnapi')
    parser.add_argument('--ndk', type=Path)
    parser.add_argument('--iterations', type=int, help='default 20; candidate frames default 160')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--include-cpu', action='store_true', help='explicitly include slow reference CPU results')
    parser.add_argument('--fp32', action='store_true', help='NNAPI only: disable optional FP16 relaxation')
    parser.add_argument('--build-only', action='store_true')
    parser.add_argument('--bundle-manifest', type=Path, help='candidate only: student_export.py manifest.json')
    parser.add_argument('--candidate-precision', choices=['fp16-storage', 'fp32'],
                        help='candidate only; default fp16-storage, both disable fp16 arithmetic')
    parser.add_argument('--output', type=Path, default=REPO / 'reports/mobile')
    args = parser.parse_args()
    candidate = args.backend == 'candidate'
    if args.bundle_manifest and not candidate:
        parser.error('--bundle-manifest applies only to candidate')
    if args.candidate_precision and not candidate:
        parser.error('--candidate-precision applies only to candidate')
    if candidate and args.candidate_precision is None:
        args.candidate_precision = 'fp16-storage'
    if args.iterations is None:
        args.iterations = 160 if candidate else 20
    minimum, maximum = (8, 160) if candidate else (1, 100)
    if not minimum <= args.iterations <= maximum or not 0 <= args.warmup <= 20:
        parser.error(f'iterations must be {minimum}..{maximum} and warmup 0..20')
    if args.backend in ('vulkan', 'ncnn', 'candidate') and args.include_cpu:
        parser.error('GPU probes do not benchmark CPU implementations; omit --include-cpu')
    if args.backend != 'nnapi' and args.fp32:
        parser.error('--fp32 applies only to NNAPI; ncnn always compares both precisions')
    token = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    report_dir = args.output / f'benchmark_{token}'
    report_dir.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'stage': 'build', 'created_at_utc': token, 'backend': args.backend,
              'limitations': [
                  'NNAPI is a deprecated driver diagnostic; Vulkan uses a custom kernel, not an optimized ML runtime.',
                  'Random FP32 small operator probes; no attention/quantization/audio/VAE model.',
                  'No CPU fallback for Vulkan. NNAPI specifies one device; its vendor internals remain opaque.',
                  'ADB shell process memory is not APK memory and excludes some driver allocations.',
                  'No sustained thermal test, full-model latency, phone video FPS or real-time conclusion.']}
    if args.backend == 'ncnn':
        report['limitations'] = [
            'Random-weight FFN, self-attention without mask/RoPE, and 2D convolution decoder proxy; no trained video model.',
            'GPU layers use Vulkan interfaces exclusively; FP32 CPU calculations only validate numerical output.',
            'Wall latency includes recording, input upload, GPU execution, output download and synchronization.',
            'FP16 storage is observed; arithmetic option is requested, not a disassembly-level instruction audit.',
            'Pipeline creation and weight upload are reported separately; no APK, sustained thermal test or video FPS.',
            'Convolution3D factory availability does not establish Wan VAE conversion support.']
    if candidate:
        report['candidate_precision'] = args.candidate_precision
        report['limitations'] = [
            'Architecture candidate with random or checkpoint-loaded weights; no video quality conclusion or Wan export.',
            'Synthetic image and audio feature inputs; no WAV frontend, image file decoder, player or encoder.',
            'Single forward per RGB frame, cached identity and GPU recurrent state; no diffusion or 3D VAE.',
            'Every RGB/state output checked finite; first eight frames compared to the same FP32 CPU graph.',
            'Frame latency includes audio upload, graph recording, inference, RGB/state download and sync.',
            'Cold latency starts inside native main, excludes OS launch; optional bundle reading is included.',
            'Vulkan driver/shader caches are uncontrolled; cold field means a new native process, not a cleared-cache device.',
            'GPU phase RSS includes host allocations and excludes some driver memory; CPU validation follows measurement.',
            'Back-to-back throughput has no playback pacing and is not a 10-minute thermal or video quality test.']
    try:
        exported = None
        if args.bundle_manifest:
            exported, bundle = load_candidate_export(args.bundle_manifest)
            if args.iterations > exported['frames']:
                raise ValueError('requested more frames than the export contains')
            report['candidate_export'] = exported
        binary, compilation = build(REPO / 'build/mobile' / args.backend, args.ndk, args.backend)
        if exported:
            compilation['artifacts'].append(str(bundle))
            compilation['artifact_sha256'][bundle.name] = exported['bundle_sha256']
        report['build'] = compilation
        shutil.copyfile(binary.parent / 'build.log', report_dir / 'build.log')
        print(f'Built: {binary}', flush=True)
        if args.build_only:
            report['status'] = 'build_only'
            return 0
        adb = shutil.which('adb')
        if not adb:
            raise ValueError('adb not on PATH')
        serial = choose_device(devices(execute([adb, 'devices'])), args.serial)
        base = [adb, '-s', serial]
        report.update(stage='device_run', serial=serial)
        report['model'] = execute(base + ['shell', 'getprop', 'ro.product.model'])
        report['soc'] = execute(base + ['shell', 'getprop', 'ro.soc.model'])
        report['memory_before'] = execute(base + ['shell', 'cat', '/proc/meminfo'])
        remote = f'/data/local/tmp/nova-avatar-bench/{token}'
        report['device_directory'] = remote
        execute(base + ['shell', 'mkdir', '-p', remote])
        execute(base + ['push', str(binary), remote + '/' + binary.name])
        execute(base + ['shell', 'chmod', '700', remote + '/' + binary.name])
        for artifact in compilation['artifacts']:
            execute(base + ['push', artifact, remote + '/' + Path(artifact).name])
        arguments = ['--iterations', str(args.iterations), '--warmup', str(args.warmup)]
        if candidate:
            arguments.extend(['--precision-mode', '0' if args.candidate_precision == 'fp32' else '2'])
        if args.include_cpu:
            arguments.append('--include-cpu')
        if args.fp32 and args.backend == 'nnapi':
            arguments.append('--fp32')
        if args.backend == 'vulkan':
            arguments.extend(['--shader', remote + '/matmul.spv'])
        if exported:
            arguments.extend(['--bundle', remote + '/candidate.bundle'])
            if exported.get('provenance', {}).get('teacher_trained') is True:
                arguments.extend(['--teacher-trained', '1'])
        # Android timeout bounds the device process, not only the host adb client.
        command = base + ['shell']
        if args.backend in ('ncnn', 'candidate'):
            command += ['env', 'LD_LIBRARY_PATH=' + remote]
        device_timeout = 300 if candidate else 180
        command += ['timeout', str(device_timeout), remote + '/' + binary.name] + arguments
        result = subprocess.run(command, capture_output=True, text=True, timeout=device_timeout + 20)
        (report_dir / 'device.stdout').write_text(result.stdout)
        (report_dir / 'device.stderr').write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(f'device probe failed ({result.returncode}): {result.stderr.strip()}')
        native = json.loads(result.stdout)
        report['native'] = native
        report.update(summarize_candidate(native) if candidate else summarize(native))
        if exported and not report['pytorch_equivalence_validated']:
            report['status'] = 'candidate_graph_failed'
        report['memory_after'] = execute(base + ['shell', 'cat', '/proc/meminfo'])
        print(json.dumps(report['native'], indent=2))
        print(f"Result: {report['status']}")
        # Diagnostic success does not imply any usable hardware acceleration.
        return 1 if candidate and report['status'] == 'candidate_graph_failed' else 0
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        report.update(status='failed', error=str(exc))
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    finally:
        destination = report_dir / 'report.json'
        destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'Report: {destination.resolve()}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
