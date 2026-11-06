"""Test a causal audio-to-lip predictor on a separate clip of the same person."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.student_local_probe import LocalRenderer
from scripts.student_mouth_detail_probe import BOX, HEIGHT, WIDTH, guide_for, read_frames, to_pixels, to_tensor
from scripts.student_pose_probe import point_xy
from scripts.student_train import write_video
from student.audio import features
from student.data import audio_timeline, load, sha256

HISTORY = 16
HOP_SAMPLES = 160
RATE = 16000
OUTER_POINTS = 14
INNER_POINTS = 6
SHAPE = (OUTER_POINTS+INNER_POINTS, 2)


def causal_windows(waveform, times):
    ends = np.rint(np.asarray(times)*RATE).astype(np.int64)
    windows = np.maximum(0, ends[:,None] - np.arange(HISTORY-1,-1,-1)*HOP_SAMPLES)
    unique, inverse = np.unique(windows,return_inverse=True)
    return torch.from_numpy(features(waveform,unique/RATE)[inverse].reshape(len(ends),HISTORY,40))


def read_lips(path, count):
    raw = json.loads(path.read_text())
    names = ['reference.png', *(f'{i:03}.png' for i in range(count))]
    rows = {row['file']:row for row in raw}
    if len(raw) != len(names) or set(rows) != set(names):
        raise ValueError('expected a matched reference and one lip contour per frame')
    result = []
    for name in names:
        row = rows[name]
        if len(row['outerLips']) != OUTER_POINTS or len(row['innerLips']) != INNER_POINTS:
            raise ValueError(f'incompatible lip contour in {name}')
        result.append([point_xy(point) for point in row['outerLips']+row['innerLips']])
    return torch.tensor(np.asarray(result),dtype=torch.float32)


def opening(points):
    inner = points[...,OUTER_POINTS:,1]
    return inner.amax(dim=-1)-inner.amin(dim=-1)


class AudioLip(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio = nn.Sequential(nn.Conv1d(40,48,3,padding=1),nn.SiLU(),
                                   nn.Conv1d(48,48,3,padding=1),nn.SiLU(),nn.Flatten())
        self.output = nn.Sequential(nn.Linear(48*HISTORY,96),nn.SiLU(),nn.Linear(96,SHAPE[0]*SHAPE[1]))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, windows):
        return self.output(self.audio(windows.transpose(1,2))).reshape(-1,*SHAPE)


def metrics(points, target):
    gap, truth = opening(points),opening(target)
    delta = gap-truth
    active = truth >= torch.quantile(truth,.75)
    correlation = None
    if gap.std() > 1e-6 and truth.std() > 1e-6:
        correlation = float(torch.corrcoef(torch.stack((gap,truth)))[0,1])
    return {'landmark_mae_px_256':float((points-target).abs().mean()),
            'opening_mae_px_256':float(delta.abs().mean()),
            'openest_quartile_mae_px_256':float(delta[active].abs().mean()),
            'opening_std_px_256':float(gap.std()),'opening_correlation':correlation}


def renderer_rows(points):
    def item(point):
        x,y = point
        return {'x':float(np.clip((x-32)/192,0,1)),'y':float(np.clip(y/256,0,1))}
    return [{'outerLips':[item(p) for p in shape[:OUTER_POINTS]],
             'innerLips':[item(p) for p in shape[OUTER_POINTS:]]} for shape in points.numpy()]


def render_static_reference(reference_path, checkpoint, point_sequences):
    with Image.open(reference_path) as image:
        reference = ImageOps.exif_transpose(image).convert('RGB').resize((WIDTH,HEIGHT),Image.Resampling.LANCZOS)
    renderer = LocalRenderer(4).eval()
    weights = torch.load(checkpoint,map_location='cpu',weights_only=True)
    if weights.get('architecture') != 'native_lip_detail_probe' or tuple(weights.get('box',())) != BOX:
        raise ValueError('incompatible mouth detail checkpoint')
    renderer.load_state_dict(weights['state_dict'],strict=True)
    rgb = to_tensor([reference.crop(BOX)])
    output = {}
    x0,y0,x1,y1 = BOX
    with torch.inference_mode():
        for name,points in point_sequences.items():
            guide = torch.from_numpy(np.stack([guide_for(row) for row in renderer_rows(points)]))
            alpha = F.avg_pool2d(F.max_pool2d(guide[:,0:1],23,1,11),9,1,4,count_include_pad=False)
            patches = torch.cat([renderer(rgb,rgb.expand(len(g),-1,-1,-1),g,a)
                                 for g,a in zip(guide.split(8),alpha.split(8))])
            if not torch.isfinite(patches).all():
                raise RuntimeError('nonfinite mouth preview')
            frames = np.broadcast_to(np.asarray(reference),(len(points),HEIGHT,WIDTH,3)).copy()
            frames[:,y0:y1,x0:x1] = to_pixels(patches)
            output[name] = frames
    return output


def run(args):
    torch.set_num_threads(2)
    torch.manual_seed(713)
    record, data = load(args.dataset)
    if sha256(args.reference) != record['source_reference_sha256']:
        raise ValueError('reference must match the reviewed training sample')
    if sha256(args.heldout_video) == record['source_video_sha256']:
        raise ValueError('held-out clip must differ from the training clip')
    count = record['frames']
    target_images = read_frames(args.heldout_video,data['times'],(WIDTH,HEIGHT),allow_tail=True)
    heldout_audio, alignment = audio_timeline(args.heldout_video,0,count*1000)
    if alignment['contiguous_audio_samples'] < len(heldout_audio) or np.array_equal(heldout_audio,data['audio']):
        raise ValueError('held-out audio is missing, incomplete, or duplicates training audio')
    train_lips = read_lips(args.train_tracks,count)
    test_lips = read_lips(args.heldout_tracks,count)
    if (train_lips[0]-test_lips[0]).abs().max() > .5:
        raise ValueError('lip tracks do not use the same reference image')
    base = train_lips[0]
    labels = train_lips[1:]
    windows = causal_windows(data['audio'],data['times'])
    model = AudioLip()
    opt = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    rng = np.random.default_rng(713)
    active = np.flatnonzero(opening(labels).numpy()>=float(torch.quantile(opening(labels),.75)))
    for step in range(1,args.steps+1):
        batch = np.concatenate((rng.choice(active,8),rng.integers(0,count,8)))
        earlier = np.maximum(0,batch-1)
        both = np.concatenate((batch,earlier))
        opt.zero_grad(set_to_none=True)
        predicted = model(windows[both])*16 + base
        delta = (predicted-labels[both])/16
        aperture_error = (opening(predicted)-opening(labels[both]))/16
        loss = (F.smooth_l1_loss(delta,torch.zeros_like(delta),beta=.12)
                + .7*aperture_error.abs().mean() + .25*(delta[:16]-delta[16:]).abs().mean())
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite training loss')
        loss.backward()
        opt.step()
        if step in (1,args.steps) or step%100==0:
            print(f'step={step} loss={float(loss.detach()):.6f}',flush=True)
    with torch.inference_mode():
        prediction = model(windows)*16 + base
        test_windows = causal_windows(heldout_audio,data['times'])
        heldout = model(test_windows)*16 + base
        shifted = model(torch.roll(test_windows,count//2,dims=0))*16 + base
        silence = model(torch.full_like(test_windows,-1))*16 + base
    targets = test_lips[1:]
    if not all(torch.isfinite(points).all() for points in (prediction,heldout,shifted,silence)):
        raise RuntimeError('nonfinite lip prediction')
    report = {'training_dataset_manifest_sha256':sha256(args.dataset),
              'train_track_sha256':sha256(args.train_tracks),'heldout_track_sha256':sha256(args.heldout_tracks),
              'heldout_video_sha256':sha256(args.heldout_video),'steps':args.steps,'frames_per_clip':count,
              'parameters':sum(p.numel() for p in model.parameters()),
              'audio_frontend':{'features':'causal_logmel40_v1','hop_samples':HOP_SAMPLES,'context_windows':HISTORY,
                                'target_pts_is_final_window_end':True},
              'training_clip':metrics(prediction,labels),
              'heldout_same_person':{'audio':metrics(heldout,targets),'reference':metrics(base.expand_as(targets),targets),
                                     'training_mean':metrics(labels.mean(0).expand_as(targets),targets),
                                     'shifted_audio':metrics(shifted,targets),'silence':metrics(silence,targets)},
              'reference_only_inference_preview':True,'independent_identities_validated':False,
              'teacher_video_is_lip_motion_reference_not_sync_ground_truth':True,'mobile_verified':False}
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    torch.save({'schema_version':1,'architecture':'causal_audio_lip_probe','state_dict':model.state_dict(),
                'reference_lips':base,'audio_frontend':report['audio_frontend']},args.output/'checkpoint.pt')
    if args.mouth_checkpoint:
        variants = render_static_reference(args.reference,args.mouth_checkpoint,{'audio':heldout,'oracle':targets})
        write_video(args.output/'normal_audio_only.mp4',variants['audio'],heldout_audio)
        teacher = np.stack([np.asarray(image) for image in target_images])
        write_video(args.output/'normal_comparison.mp4',np.concatenate((teacher,variants['audio'],variants['oracle']),axis=2),heldout_audio)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','reference','train-tracks','heldout-video','heldout-tracks','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--mouth-checkpoint',type=Path)
    parser.add_argument('--steps',type=int,default=400)
    args = parser.parse_args()
    if not 1 <= args.steps <= 1000:
        parser.error('--steps must be 1..1000')
    run(args)
