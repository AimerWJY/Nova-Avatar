"""Load the actual converted v2 graphs and validate against PyTorch on Android."""
import argparse
import json
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from mobile_benchmark import build, REPO
from mobile_profile import choose_device, devices, execute
from mobile_runtime import digest


def assets(model_manifest, fixture_manifest):
    model = json.loads(model_manifest.read_text()); fixture = json.loads(fixture_manifest.read_text())
    if (model.get('architecture') != 'mouth_refinement_v2' or not model.get('ncnn_conversion', {}).get('passed')
            or fixture.get('format') != 'converted_v2_fixture_v1' or fixture.get('schema_version') != 1
            or fixture.get('model_manifest_sha256') != digest(model_manifest)
            or fixture.get('checkpoint_sha256') != model.get('checkpoint_sha256')
            or fixture.get('dataset_manifest_sha256') != model.get('dataset_manifest_sha256')
            or fixture.get('mobile_output_policy', {'version':'raw_v2'}) != model.get('mobile_output_policy', {'version':'raw_v2'})
            or fixture.get('fixture_file') != 'v2.fixture' or type(fixture.get('frames')) is not int
            or not 8 <= fixture['frames'] <= 160):
        raise ValueError('converted model and reference fixture provenance mismatch')
    files = graph_files(model_manifest, model)
    path = fixture_manifest.resolve().parent/'v2.fixture'
    if (path.is_symlink() or not path.is_file() or path.stat().st_size > 128*1024*1024
            or path.stat().st_size != fixture.get('fixture_bytes') or digest(path) != fixture.get('fixture_sha256')):
        raise ValueError('v2 fixture size/SHA256 mismatch')
    files.append(path)
    return model, fixture, files


def graph_files(model_manifest, model):
    graphs = {row['name']: row for row in model['ncnn_conversion']['graphs']}
    if set(graphs) != {'encoder', 'frame'}:
        raise ValueError('requires both encoder and frame graphs')
    files = []
    for name in ('encoder', 'frame'):
        for extension, field in (('param', 'param_sha256'), ('bin', 'bin_sha256')):
            path = model_manifest.resolve().parent/f'{name}.ncnn.{extension}'
            if path.is_symlink() or not path.is_file() or digest(path) != graphs[name][field]:
                raise ValueError('converted graph SHA256 mismatch')
            files.append(path)
    return files


def summarize(native):
    valid = (native.get('status') == 'numerical_validation_passed'
             and native.get('all_outputs_finite') is True and native.get('max_scaled_error', float('inf')) <= 1
             and native.get('state_reset_max_scaled_error', float('inf')) <= 1)
    if not valid:
        status = 'numerical_validation_failed'
    elif not native['vulkan_requested']:
        status = 'validated_cpu'
    elif native['compute_layers_without_vulkan_support']:
        status = 'validated_with_cpu_fallback'
    else:
        status = 'validated_vulkan_capable_graph'
    return {'status': status, 'phone_model_numerics_validated': valid,
            'phone_model_run_completed': native.get('all_outputs_finite') is True,
            'warm_compute_16fps_budget_met': bool(valid and native.get('diagnostic_frame', -1) < 0 and native['p95_ms'] <= 62.5),
            'gpu_only_execution_verified': False, 'video_quality_validated': False,
            'end_to_end_first_playback_verified': False}


def run_probe(command, output, name):
    result = subprocess.run(command, capture_output=True, text=True, timeout=320)
    (output/f'{name}.stdout').write_text(result.stdout)
    (output/f'{name}.stderr').write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f'{name} runtime failed ({result.returncode}): {result.stderr[-3000:]}')
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-manifest', type=Path, required=True)
    parser.add_argument('--fixture-manifest', type=Path, required=True)
    parser.add_argument('--serial')
    parser.add_argument('--device', choices=['vulkan', 'cpu'], default='vulkan')
    parser.add_argument('--frames', type=int)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--build-only', action='store_true')
    parser.add_argument('--diagnostic-frame', type=int, help='dump intermediate tensors; diagnostic runs are not performance acceptance')
    parser.add_argument('--compare-pipeline-cache', action='store_true',
                        help='compare Vulkan startup with and without a cache generated on the same device')
    args = parser.parse_args()
    if not 0 <= args.warmup <= 20 or (args.frames is not None and not 8 <= args.frames <= 160):
        parser.error('warmup 0..20, frames 8..160 required')
    if args.diagnostic_frame is not None and not 0 <= args.diagnostic_frame < 160:
        parser.error('diagnostic frame must be 0..159')
    if args.compare_pipeline_cache and (args.device != 'vulkan' or args.diagnostic_frame is not None):
        parser.error('pipeline cache comparison requires Vulkan without diagnostics')
    token = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    output = REPO/'reports/mobile'/('model_'+token); output.mkdir(parents=True)
    report = {'status': 'running', 'device_request': args.device, 'limitations': [
        'Actual trained v2 weights and precomputed training-clip features; no independent quality acceptance.',
        'Vulkan is requested through ncnn Net; layers without Vulkan capability may fall back to CPU.',
        'Capability flags do not constitute a GPU instruction or runtime placement trace.',
        'CPU Mat inputs/outputs include per-frame upload/download of image, caches, audio and recurrent state.',
        'Frame latency excludes comparison; stream wall time includes full-output validation.',
        'Fixture read, model load and setup count toward native first output; OS/library launch and player do not.',
        'Resident first RGB re-encodes the fixture image with models already loaded and precomputed audio features.',
        'No phone audio frontend, APK, playback, cleared shader cache or sustained thermal test.']}
    try:
        model, fixture, files = assets(args.model_manifest, args.fixture_manifest)
        count = args.frames or fixture['frames']
        if count > fixture['frames']:
            raise ValueError('requested frames exceed fixture')
        if args.diagnostic_frame is not None and args.diagnostic_frame >= count:
            raise ValueError('diagnostic frame exceeds requested frames')
        report.update(model_manifest_sha256=digest(args.model_manifest), fixture=fixture,
                      mobile_output_policy=model.get('mobile_output_policy', {'version':'raw_v2'}))
        binary, compilation = build(REPO/'build/mobile/converted', backend='converted')
        report['build'] = compilation
        shutil.copyfile(binary.parent/'build.log', output/'build.log')
        if args.build_only:
            report['status'] = 'build_only'; return 0
        adb = shutil.which('adb')
        if not adb:
            raise ValueError('ADB missing')
        serial = choose_device(devices(execute([adb, 'devices'])), args.serial)
        base = [adb, '-s', serial]; remote = '/data/local/tmp/nova-avatar-model/'+token
        report.update(serial=serial, remote_directory=remote, memory_before=execute(base+['shell','cat','/proc/meminfo']))
        execute(base+['shell','mkdir','-p',remote])
        if args.diagnostic_frame is not None:
            execute(base+['shell','mkdir','-p',remote+'/diagnostics'])
        for path in [binary, *map(Path, compilation['artifacts']), *files]:
            print('Pushing '+path.name, flush=True)
            execute(base+['push', str(path), remote+'/'+path.name], timeout=120)
        execute(base+['shell','chmod','700',remote+'/'+binary.name])
        command = base+['shell','env','LD_LIBRARY_PATH='+remote,'timeout','300',remote+'/'+binary.name,
                        '--directory',remote,'--fixture',remote+'/v2.fixture','--frames',str(count),
                        '--warmup',str(args.warmup),'--gpu','1' if args.device=='vulkan' else '0']
        if args.diagnostic_frame is not None:
            command += ['--diagnostic-frame', str(args.diagnostic_frame), '--diagnostic-dir', remote+'/diagnostics']
        if args.compare_pipeline_cache:
            comparison = {'driver_cache_controlled': False, 'runs': {}}
            report['pipeline_cache_comparison'] = comparison
            # Each invocation starts a fresh process. Keep the generated cache in this run's device directory.
            for mode in ('save', 'restore', 'off'):
                print(f'Running Vulkan startup probe: {mode}', flush=True)
                extra = [] if mode == 'off' else ['--pipeline-cache', mode]
                native = run_probe(command+extra, output, 'device_'+mode)
                comparison['runs'][mode] = native
                if not summarize(native)['phone_model_numerics_validated']:
                    raise RuntimeError(f'{mode} startup run did not pass numerical validation')
            native = comparison['runs']['restore']
        else:
            native = run_probe(command, output, 'device')
        report['native'] = native; report.update(summarize(native))
        if args.diagnostic_frame is not None:
            execute(base+['pull', remote+'/diagnostics', str(output/'diagnostics')], timeout=120)
        report['memory_after'] = execute(base+['shell','cat','/proc/meminfo'])
        console = dict(native)
        if not native['vulkan_requested']:
            console['layers_without_vulkan_support'] = '(see report.json; CPU-mode flags are not GPU capability tests)'
        print(json.dumps(console, indent=2))
        return 0 if report['phone_model_numerics_validated'] else 1
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        report.update(status='failed', error=str(exc)); print(str(exc), file=sys.stderr); return 1
    finally:
        (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print('Report: '+str(output/'report.json'), flush=True)


if __name__ == '__main__':
    sys.exit(main())
