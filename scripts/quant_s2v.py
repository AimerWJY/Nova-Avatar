"""Run a pinned, offline Wan2.2-S2V-14B Q2_K short-clip benchmark on CUDA."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from infer_s2v import FPS, NEGATIVE, PROMPT, probe_audio

REPO = Path(__file__).resolve().parents[1]
PIN = json.loads((REPO/'configs/wan-s2v-q2.json').read_text())
UPSTREAM = 'https://github.com/leejet/stable-diffusion.cpp.git'
WIDTH, HEIGHT = 480, 640
REQUESTED_FRAMES = 33
GPU_QUERY = ['nvidia-smi','--query-gpu=uuid,name,memory.used,memory.total',
             '--format=csv,noheader,nounits']


def command(*args, **kwargs):
    return subprocess.run([str(value) for value in args],check=True,**kwargs)


def capture(*args):
    return command(*args,capture_output=True,text=True,timeout=30).stdout.strip()


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def require_tools(*names):
    missing = [name for name in names if not shutil.which(name)]
    if missing:
        raise ValueError('missing tools: '+', '.join(missing))


def paths_for(directory):
    return {key:directory/spec['url'].rsplit('/',1)[-1] for key,spec in PIN['assets'].items()}


def verify(path,spec):
    if (not path.is_file() or path.stat().st_size != spec['bytes'] or sha256(path) != spec['sha256']):
        raise ValueError(f'invalid size/SHA256: {path}. Move the invalid file aside before retrying.')


def download(directory):
    require_tools('curl')
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.download.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for key,path in paths_for(directory).items():
            spec = PIN['assets'][key]
            if path.exists():
                verify(path,spec)
                print(f'Verified: {path.name}',flush=True)
                continue
            partial = path.with_name(path.name+'.partial')
            current = partial.stat().st_size if partial.exists() else 0
            if shutil.disk_usage(directory).free < spec['bytes']-current:
                raise ValueError(f'insufficient space for {path.name}')
            if current < spec['bytes']:
                command('curl','--fail','--location','--retry','3','--connect-timeout','20',
                        '--speed-limit','1024','--speed-time','120','--continue-at','-',
                        '--output',partial,spec['url'])
            verify(partial,spec)
            partial.rename(path)
            print(f'Verified: {path.name}',flush=True)


def check_cli(binary):
    if not binary.is_file() or not os.access(binary,os.X_OK):
        raise ValueError(f'sd-cli unavailable: {binary}; run quant-s2v build first')
    process = command(binary,'--help',capture_output=True,text=True,timeout=30)
    help_text = process.stdout+process.stderr
    for flag in ('--audio-encoder','--audio','--video-frames','--output-begin-idx','--backend','--offload-to-cpu'):
        if flag not in help_text:
            raise ValueError(f'sd-cli lacks required flag: {flag}')


def build(work,jobs):
    require_tools('cmake','git','nvidia-smi')
    nvcc = shutil.which('nvcc')
    if not nvcc:
        toolkit = Path(os.environ.get('CUDA_HOME','/usr/local/cuda'))/'bin/nvcc'
        if toolkit.is_file():
            nvcc = str(toolkit)
    if not nvcc:
        raise ValueError('CUDA Toolkit compiler (nvcc) missing. Use a CUDA development image or pass a CUDA sd-cli with --binary for run.')
    source = work/'source'
    work.mkdir(parents=True,exist_ok=True)
    with (work/'.build.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not source.exists():
            command('git','clone','--depth','1','--branch',PIN['runtime_tag'],UPSTREAM,source)
        if capture('git','-C',source,'rev-parse','HEAD') != PIN['runtime_commit']:
            raise ValueError(f'runtime checkout has the wrong commit: {source}')
        if capture('git','-C',source,'status','--porcelain','--untracked-files=no'):
            raise ValueError(f'runtime checkout contains tracked edits: {source}')
        command('git','-C',source,'submodule','update','--init','--recursive','--depth','1')
        binary_dir = work/'build'
        command('cmake','-S',source,'-B',binary_dir,'-DCMAKE_BUILD_TYPE=Release','-DSD_CUDA=ON',
                '-DCMAKE_CUDA_ARCHITECTURES=89',f'-DCMAKE_CUDA_COMPILER={nvcc}',
                '-DSD_WEBP=OFF','-DSD_WEBM=OFF')
        command('cmake','--build',binary_dir,'--target','sd-cli','--parallel',jobs)
        binary = binary_dir/'bin/sd-cli'
        check_cli(binary)
        print(f'Built: {binary}',flush=True)


def arguments(binary,assets,output):
    return [str(binary),'-M','vid_gen',
            *[value for key,path in assets.items() for value in ('--'+key,str(path))],
            '--backend','cuda0','--offload-to-cpu','--vae-tiling',
            '--sampling-method','euler','--steps','20','--cfg-scale','6.0','--flow-shift','3.0',
            '--seed','42','-W',str(WIDTH),'-H',str(HEIGHT),'--video-frames',str(REQUESTED_FRAMES),
            '-p',PROMPT,'-n',NEGATIVE,'-i',str(output/'reference.png'),
            '--audio',str(output/'audio.wav'),'--output-begin-idx','0',
            '-o',str(output/'frames/frame_%04d.png'),'-v']


def gpu_summary(csv_path):
    samples = {}
    for row in csv.reader(csv_path.read_text().splitlines()):
        if len(row) != 4:
            continue
        uuid_value,name,used,total = (value.strip() for value in row)
        if used.isdecimal() and total.isdecimal():
            item = samples.setdefault(uuid_value,{'name':name,'total_mib':int(total),
                                                  'sampled_peak_used_mib':0,'samples':0})
            item['sampled_peak_used_mib'] = max(item['sampled_peak_used_mib'],int(used))
            item['samples'] += 1
    return samples


def run(args,assets,binary):
    require_tools('ffmpeg','ffprobe','nvidia-smi')
    check_cli(binary)
    for key,path in assets.items():
        verify(path,PIN['assets'][key])
    image = (args.image or args.root/'inputs/person_01.png').resolve(strict=True)
    audio = (args.audio or args.root/'inputs/emotion.wav').resolve(strict=True)
    if probe_audio(audio)['duration_seconds'] < REQUESTED_FRAMES/FPS:
        raise ValueError(f'audio must cover {REQUESTED_FRAMES/FPS:.4f} seconds')
    token = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    output = args.root/'outputs'/('s2v_q2_'+token)
    output.mkdir(parents=True)
    gpu_log = output/'gpu.csv'
    cli = arguments(binary,assets,output)
    report = {'status':'running','variant':'Wan2.2-S2V-14B-Q2_K','phase':'cuda_short_clip',
              'runtime_expected_commit':PIN['runtime_commit'],'binary_sha256':sha256(binary),
              'binary_override':args.binary is not None,'command':cli,
              'source_image':str(image),'source_audio':str(audio),
              'source_image_sha256':sha256(image),'source_audio_sha256':sha256(audio),
              'assets':PIN['assets'],'phone_verified':False,'quality_reviewed':False}
    try:
        report['gpu_before'] = capture(*GPU_QUERY).splitlines()
        command('ffmpeg','-nostdin','-v','error','-n','-i',image,'-frames:v','1','-vf',
                f'scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2',
                '-update','1',output/'reference.png')
        command('ffmpeg','-nostdin','-v','error','-n','-i',audio,'-t',str(REQUESTED_FRAMES/FPS),
                '-vn','-ac','1','-ar','16000','-c:a','pcm_s16le',output/'audio.wav')
        print(f'Generating; follow {output/"sd-cli.log"}',flush=True)
        started = time.monotonic()
        with gpu_log.open('w') as gpu_stream, (output/'sd-cli.log').open('w') as cli_log:
            monitor = subprocess.Popen(GPU_QUERY+['--loop-ms=500'],stdout=gpu_stream,stderr=subprocess.STDOUT)
            try:
                command(*cli,stdout=cli_log,stderr=subprocess.STDOUT)
            finally:
                monitor.terminate()
                monitor.wait(timeout=10)
                report['cli_seconds_including_load_decode_and_png'] = time.monotonic()-started
        frames = sorted((output/'frames').glob('frame_*.png'))
        if len(frames) not in (REQUESTED_FRAMES-1,REQUESTED_FRAMES) or [f.name for f in frames] != [
                f'frame_{i:04}.png' for i in range(len(frames))]:
            raise ValueError(f'unexpected generated frame sequence: {len(frames)} frames')
        video = output/'video.mp4'
        command('ffmpeg','-nostdin','-v','error','-n','-framerate',FPS,'-start_number','0',
                '-i',output/'frames/frame_%04d.png','-i',output/'audio.wav','-map','0:v:0','-map','1:a:0',
                '-c:v','libx264','-pix_fmt','yuv420p',
                '-c:a','aac','-shortest','-movflags','+faststart',video)
        info = json.loads(capture('ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',video))
        stream = next((item for item in info['streams'] if item['codec_type']=='video'),None)
        audio_stream = next((item for item in info['streams'] if item['codec_type']=='audio'),None)
        if (not stream or stream['width']!=WIDTH or stream['height']!=HEIGHT
                or Fraction(stream['avg_frame_rate'])!=FPS or stream.get('nb_read_frames')!=str(len(frames))
                or not audio_stream or float(audio_stream.get('duration',0))<(len(frames)-1)/FPS):
            raise ValueError('muxed output does not match expected frames, geometry and audio')
        report.update(status='success',video=str(video),frames=len(frames),fps=FPS,
                      duration_seconds=float(info['format']['duration']))
        print(f'Video: {video}',flush=True)
    except (OSError,ValueError,subprocess.SubprocessError) as error:
        report.update(status='failed',error=str(error))
        raise
    finally:
        if gpu_log.exists():
            report['gpu_device_wide_usage_samples'] = gpu_summary(gpu_log)
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(f'Report: {output/"report.json"}',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('plan','build','assets','run'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--assets-dir',type=Path)
    parser.add_argument('--binary',type=Path,help='optional prebuilt CUDA sd-cli; upstream commit is then user-supplied')
    parser.add_argument('--image',type=Path)
    parser.add_argument('--audio',type=Path)
    parser.add_argument('--jobs',type=int,default=4)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 16:
        parser.error('--jobs must be 1..16')
    args.root = args.root.resolve()
    work = args.root/'tools'/('stable-diffusion-cpp-'+PIN['runtime_tag'])
    directory = (args.assets_dir or args.root/'models/wan-s2v-q2').resolve()
    binary = args.binary.resolve() if args.binary else work/'build/bin/sd-cli'
    assets = paths_for(directory)
    try:
        if args.action == 'plan':
            print(json.dumps({'runtime_commit':PIN['runtime_commit'],'assets_dir':str(directory),
                              'total_weight_bytes':sum(spec['bytes'] for spec in PIN['assets'].values()),
                              'asset_paths':{key:str(path) for key,path in assets.items()},
                              'source_image':str(args.image or args.root/'inputs/person_01.png'),
                              'source_audio':str(args.audio or args.root/'inputs/emotion.wav'),
                              'command':arguments(binary,assets,args.root/'outputs/<run>')},indent=2))
        elif args.action == 'build':
            build(work,args.jobs)
        elif args.action == 'assets':
            download(directory)
        else:
            run(args,assets,binary)
    except (OSError,ValueError,subprocess.SubprocessError) as error:
        parser.exit(1,f'{error}\n')


if __name__ == '__main__':
    main()
