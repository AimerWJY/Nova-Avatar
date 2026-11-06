"""Build the Android CPU preview APK from an approved v2 export using the local SDK."""
import argparse
import json
import re
import subprocess
import zipfile
from pathlib import Path

from mobile_benchmark import REPO, build
from mobile_model import graph_files
from mobile_profile import sdk_tools
from mobile_runtime import digest


def package(manifest, output):
    model = json.loads(manifest.read_text())
    policy = model.get('mobile_output_policy', {})
    photo_hash = policy.get('source_reference_sha256', '')
    if (model.get('architecture') != 'mouth_refinement_v2'
            or model.get('image_size') != [256, 256] or model.get('audio_history_shape') != [8, 40]
            or not model.get('ncnn_conversion', {}).get('passed')
            or policy.get('version') != 'fixed_letterbox_mask_v1'
            or policy.get('content_box') != [32, 0, 224, 256]
            or not re.fullmatch('[a-f0-9]{64}', photo_hash)):
        raise ValueError('preview requires the v2 export and its verified 3:4 reference photo')
    graphs = graph_files(manifest, model)
    tools = sdk_tools()
    if not tools['sdk']:
        raise ValueError('Android SDK not found')
    sdk = Path(tools['sdk'])
    platform = sdk/'platforms/android-34/android.jar'
    sdk_bin = sdk/'build-tools/34.0.0'
    if not platform.is_file() or not sdk_bin.is_dir():
        raise ValueError('install Android platform and build tools 34.0.0')
    source = REPO/'mobile/android'
    output.mkdir(parents=True, exist_ok=False)
    classes = output/'classes'
    classes.mkdir()
    native, build_report = build(output/'native', backend='preview')

    def run(*command):
        subprocess.run([str(item) for item in command], check=True, timeout=120)

    run('javac', '--release', '8', '-classpath', platform, '-d', classes, *sorted(source.glob('*.java')))
    run(sdk_bin/'d8', '--min-api', '29', '--lib', platform, '--output', output,
        *sorted(classes.rglob('*.class')))
    raw = output/'raw.apk'
    run(sdk_bin/'aapt', 'package', '-f', '-M', source/'AndroidManifest.xml', '-I', platform, '-F', raw)
    with zipfile.ZipFile(raw, 'a', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(output/'classes.dex', 'classes.dex')
        for path in graphs:
            archive.write(path, 'assets/'+path.name, compress_type=zipfile.ZIP_STORED)
        archive.writestr('assets/reference.sha256', photo_hash)
        for path in [native, *map(Path, build_report['artifacts'])]:
            archive.write(path, 'lib/arm64-v8a/'+path.name)
    aligned = output/'aligned.apk'
    run(sdk_bin/'zipalign', '-f', '4', raw, aligned)
    # Keep a stable local debug key so builds can be installed as upgrades.
    key = output.parent/'nova-preview-debug.jks'
    if not key.is_file():
        run('keytool', '-genkeypair', '-keystore', key, '-storepass', 'android', '-keypass', 'android',
            '-alias', 'nova-preview', '-keyalg', 'RSA', '-keysize', '2048', '-validity', '10000',
            '-dname', 'CN=Nova Avatar Debug')
    apk = output/'nova-avatar-preview.apk'
    run(sdk_bin/'apksigner', 'sign', '--ks', key, '--ks-pass', 'pass:android', '--key-pass',
        'pass:android', '--out', apk, aligned)
    run(sdk_bin/'apksigner', 'verify', apk)
    (output/'manifest.json').write_text(json.dumps({
        'apk_sha256': digest(apk), 'model_manifest_sha256': digest(manifest),
        'reference_sha256': photo_hash, 'native_build': build_report,
        'android_source_sha256': {path.name: digest(path) for path in source.iterdir() if path.is_file()},
    }, indent=2)+'\n')
    print(apk)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        package(args.model_manifest, args.output)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
