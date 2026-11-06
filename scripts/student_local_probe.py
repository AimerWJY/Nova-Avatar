"""Train separate hand and mouth patch renderers on a tracked teacher clip."""
import argparse
import json
import sys
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.student_pose_probe import JOINTS, observations, point_xy
from scripts.student_train import write_video
from student.data import load, sha256

BOXES = {'right': (44, 142, 112, 256), 'left': (144, 142, 212, 256),
         'mouth': (104, 44, 152, 84)}
FINGERS = (
    ('thumb_cmc', 'thumb_mp', 'thumb_ip', 'thumb_tip'),
    *((f'{finger}_mcp', f'{finger}_pip', f'{finger}_dip', f'{finger}_tip')
      for finger in ('index', 'middle', 'ring', 'little')),
)


def crop(values, box):
    x0, y0, x1, y1 = box
    return values[..., y0:y1, x0:x1]


def paint(box, chains, width, polygon=False):
    x0, y0, x1, y1 = box
    # Supersample the guides so fractional landmark motion survives rasterization.
    size = ((x1-x0)*2, (y1-y0)*2)
    image = Image.new('L', size)
    draw = ImageDraw.Draw(image)
    for chain in chains:
        points = [((x-x0)*2, (y-y0)*2) for x,y in chain]
        if polygon and len(points) > 2:
            draw.polygon(points, fill=255)
        elif len(points) > 1:
            draw.line(points, fill=255, width=width*2)
        elif points:
            x,y = points[0]
            draw.ellipse((x-width,y-width,x+width,y+width), fill=255)
    return np.asarray(image.resize((size[0]//2,size[1]//2), Image.Resampling.BOX), dtype=np.float32)/255


def landmarks(row, part, box):
    if part == 'mouth':
        contours = [[point_xy(point) for point in row[name]] for name in ('outerLips','innerLips')]
        return np.stack([paint(box, [contour], 1, polygon=True) for contour in contours]), None
    elbow, wrist = [point_xy(row['body'][part+'_'+joint]) for joint in ('elbow','wrist')]
    hand = row.get('hands', {}).get(part, {})
    points = {key:point_xy(value) for key,value in hand.items() if value['confidence'] >= .25}
    # The two wrist estimates must identify the same hand in the crop.
    if 'wrist' not in points or np.linalg.norm(points['wrist']-wrist) > 25:
        points = {}
    palm = [points[key] for key in ('wrist','thumb_cmc','index_mcp','middle_mcp','ring_mcp','little_mcp') if key in points]
    planes = [paint(box, [[elbow,wrist]], 3), paint(box, [palm], 1, polygon=True)]
    for finger in FINGERS:
        # Preserve gaps where the detector did not resolve a finger joint.
        segments = [[points[a],points[b]] for a,b in zip(finger,finger[1:]) if a in points and b in points]
        segments += [[points[key]] for key in finger if key in points]
        planes.append(paint(box, segments, 2))
    return np.stack(planes), len(points)


class LocalRenderer(nn.Module):
    def __init__(self, guide_channels):
        super().__init__()
        inputs = guide_channels+6
        self.local = nn.Sequential(nn.Conv2d(inputs, 20, 3, padding=1), nn.SiLU())
        self.down = nn.Sequential(nn.Conv2d(20, 28, 3, stride=2, padding=1), nn.SiLU(),
                                  nn.Conv2d(28, 36, 3, stride=2, padding=1), nn.SiLU(),
                                  nn.Conv2d(36, 36, 3, dilation=2, padding=2), nn.SiLU(),
                                  nn.Conv2d(36, 28, 3, dilation=2, padding=2), nn.SiLU())
        self.output = nn.Sequential(nn.Conv2d(48, 20, 3, padding=1), nn.SiLU(),
                                    nn.Conv2d(20, 3, 3, padding=1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, reference, base, guide, feather):
        local = self.local(torch.cat((reference.expand_as(base),base,guide), dim=1))
        context = F.interpolate(self.down(local),size=base.shape[-2:],mode='bilinear',align_corners=False)
        correction = self.output(torch.cat((local,context),dim=1)).tanh()*feather
        return (base+correction).clamp(0,1)


def feather(box):
    x0,y0,x1,y1 = box
    ys = torch.arange(y1-y0)
    xs = torch.arange(x1-x0)
    across = torch.minimum(xs, x1-x0-1-xs)
    down = torch.minimum(ys, y1-y0-1-ys) if y1 < 256 else ys
    return (torch.minimum(down[:,None],across[None,:]).float()/5).clamp(0,1)[None,None]


def edge_error(error):
    return ((error[...,1:]-error[...,:-1]).abs().mean()
            +(error[...,1:,:]-error[...,:-1,:]).abs().mean())*.5


def run(args):
    torch.set_num_threads(2)
    torch.manual_seed(713)
    record,data = load(args.dataset)
    raw = json.loads(args.tracks.read_text())
    joints,mouth,_ = observations(raw,record['frames'])
    by_name = {row['file']:row for row in raw}
    rows = [by_name[f'{i:03}.png'] for i in range(record['frames'])]
    if not all('hands' in row for row in rows):
        raise ValueError('rerun Vision with --hands before training the local models')
    with av.open(str(args.base)) as stream:
        base_frames = list(stream.decode(video=0))
    if len(base_frames) != record['frames'] or any(
            frame.time is None or abs(frame.time - target_time) > .001
            for frame, target_time in zip(base_frames, data['times'])):
        raise ValueError('pose base and target frame timestamps differ')
    base_pixels = np.stack([frame.to_ndarray(format='rgb24') for frame in base_frames])
    if base_pixels.shape != data['frames'].shape:
        raise ValueError('pose base and target frame dimensions differ')
    reference = torch.from_numpy(data['reference']).permute(2,0,1)[None].float()/255
    base = torch.from_numpy(base_pixels).permute(0,3,1,2).float()/255
    target = torch.from_numpy(data['frames']).permute(0,3,1,2).float()/255
    args.output.mkdir(parents=True,exist_ok=False)
    composite = base.clone()
    results,weights = {},{}
    rng = np.random.default_rng(713)
    for part,box in BOXES.items():
        guides,coverage = zip(*(landmarks(row,part,box) for row in rows))
        guide = torch.from_numpy(np.stack(guides))
        original,previous,truth = (crop(value,box) for value in (reference,base,target))
        border = feather(box)
        model = LocalRenderer(guide.shape[1])
        opt = torch.optim.Adam(model.parameters(),lr=.001)
        if part == 'mouth':
            movement = mouth[1:,3]
            focus = 1+2*guide[:,0:1]+4*guide[:,1:2]
        else:
            joint = JOINTS.index(part+'_wrist')
            movement = np.linalg.norm(joints[1:,joint]-joints[0,joint],axis=-1)
            focus = 1+2*guide[:,1:].amax(dim=1,keepdim=True)
        focus = focus + (truth-previous).abs().amax(dim=1,keepdim=True)*3
        active = np.flatnonzero(movement >= np.quantile(movement,.75))
        for step in range(1,args.steps+1):
            batch = np.concatenate((rng.choice(active,2),rng.integers(0,len(rows),2)))
            earlier = np.maximum(0,batch-1)
            both = np.concatenate((batch,earlier))
            opt.zero_grad(set_to_none=True)
            predicted = model(original,previous[both],guide[both],border)
            error = predicted-truth[both]
            loss = ((error.abs()*focus[both]).mean() + .5*edge_error(error)
                    + .35*(error[:len(batch)]-error[len(batch):]).abs().mean())
            if not torch.isfinite(loss):
                raise RuntimeError(f'nonfinite {part} training loss')
            loss.backward()
            opt.step()
            if step == 1 or step%100 == 0 or step == args.steps:
                print(f'{part} step={step} loss={float(loss.detach()):.6f}',flush=True)
        with torch.inference_mode():
            predicted = torch.cat([model(original,p,g,border) for p,g in zip(previous.split(8),guide.split(8))])
        crop(composite,box).copy_(predicted)
        def score(pixels):
            error = pixels-truth
            return {'mae':float(error.abs().mean()),'active_mae':float(error[active].abs().mean()),
                    'edge_mae':float(edge_error(error))}
        results[part] = {'box':box,'previous':score(previous),'local':score(predicted),
                         'active_frame_count':len(active)}
        if part != 'mouth':
            results[part]['frames_with_10_or_more_hand_points'] = sum(count>=10 for count in coverage)
            results[part]['frames_with_no_usable_hand_points'] = coverage.count(0)
        weights[part] = model.state_dict()

    def pixels(rgb):
        return rgb.permute(0,2,3,1).mul(255).round().byte().numpy()
    output_pixels = pixels(composite)
    outside = np.ones((256,256), dtype=bool)
    for x0,y0,x1,y1 in BOXES.values():
        outside[y0:y1,x0:x1] = False
    if not np.array_equal(output_pixels[:,outside],base_pixels[:,outside]):
        raise RuntimeError('pixels changed outside the three local regions')
    write_video(args.output/'student.mp4',output_pixels,data['audio'])
    write_video(args.output/'comparison.mp4',np.concatenate((data['frames'],base_pixels,output_pixels),axis=2),data['audio'])
    torch.save({'schema_version':1,'architecture':'local_pose_patch_probe_v1','boxes':BOXES,
                'state_dicts':weights},args.output/'checkpoint.pt')
    report = {'dataset_manifest_sha256':sha256(args.dataset),'base_video_sha256':sha256(args.base),
              'raw_track_sha256':sha256(args.tracks),'frames':len(rows),'steps_per_patch':args.steps,
              'regions':results,'outside_patches_max_change':0,'same_clip_diagnostic':True,
              'uses_teacher_landmarks':True,'audio_to_pose_verified':False,'mobile_verified':False}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','base','tracks','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--steps',type=int,default=300)
    args=parser.parse_args()
    if not 1 <= args.steps <= 1000:
        parser.error('--steps must be 1..1000')
    run(args)
