"""Refine lips using native-resolution teacher crops and explicit lip contours."""
import argparse
import json
import sys
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.student_local_probe import BOXES, LocalRenderer, crop, paint
from scripts.student_pose_probe import observations, point_xy
from scripts.student_train import write_video
from student.data import load, sha256

WIDTH, HEIGHT = 480, 640
CONTENT = (32, 0, 224, 256)
SCALE = 2.5
BOX = tuple(round((value-(32 if i%2==0 else 0))*SCALE) for i,value in enumerate(BOXES['mouth']))
SAMPLES = (0, 6, 10, 14, 18, 24, 40, 64)


def read_frames(path, times, size, allow_tail=False):
    with av.open(str(path)) as container:
        frames = list(container.decode(video=0))
    count = len(times)
    if len(frames) < count or (not allow_tail and len(frames) != count):
        raise ValueError('frame count differs from the reviewed sample')
    for frame, time in zip(frames,times):
        if (frame.width,frame.height) != size or frame.time is None or abs(frame.time-time) > .001:
            raise ValueError('frame geometry or timing differs from the reviewed sample')
    return [frame.to_image() for frame in frames[:count]]


def guide_for(row):
    outer, inner = ([point_xy(p)*SCALE - [80,0] for p in row[key]] for key in ('outerLips','innerLips'))
    planes = []
    for contour in (outer,inner):
        planes.append(paint(BOX,[contour],1,polygon=True))
        planes.append(paint(BOX,[contour+contour[:1]],2))
    return np.stack(planes)


def to_tensor(images):
    return torch.from_numpy(np.stack([np.asarray(image,dtype=np.uint8) for image in images])).permute(0,3,1,2).float()/255


def to_pixels(tensor):
    return tensor.permute(0,2,3,1).mul(255).round().byte().numpy()


def masked_mean(value, weight):
    # Divide by spatial weight only after reducing the RGB channels.
    return (value.mean(1,keepdim=True)*weight).sum()/weight.sum().clamp_min(1)


def edge_error(delta, weight):
    horizontal = (delta[...,1:]-delta[...,:-1]).abs()
    vertical = (delta[...,1:,:]-delta[...,:-1,:]).abs()
    return (masked_mean(horizontal,weight[...,1:])+masked_mean(vertical,weight[...,1:,:]))*.5


def review_image(path, rows):
    board = Image.new('RGB',(4*180,len(rows)*2*165),'#252525')
    label = ImageDraw.Draw(board)
    for slot,index in enumerate(SAMPLES):
        col,group = slot%4,slot//4
        for row,(name,images) in enumerate(rows):
            x,y = col*180,(group*len(rows)+row)*165
            frame = Image.fromarray(images[index]).crop(BOX).resize((168,140),Image.Resampling.LANCZOS)
            board.paste(frame,(x+6,y+22))
            label.text((x+6,y+6),f'{name} / {index/16:.3f}s',fill='white')
    board.save(path,quality=96)


def run(args):
    torch.set_num_threads(2)
    torch.manual_seed(713)
    record,data = load(args.dataset)
    if record['frames'] <= max(SAMPLES):
        raise ValueError('diagnostic requires the full reviewed clip')
    for path,key in ((args.teacher,'source_video_sha256'),(args.reference,'source_reference_sha256')):
        if sha256(path) != record[key]:
            raise ValueError(f'{path.name} differs from the reviewed source')
    raw = json.loads(args.tracks.read_text())
    _,mouth,_ = observations(raw,record['frames'])
    rows = {row['file']:row for row in raw}
    guides = torch.from_numpy(np.stack([guide_for(rows[f'{i:03}.png']) for i in range(record['frames'])]))
    target_images = read_frames(args.teacher,data['times'],(WIDTH,HEIGHT),allow_tail=True)
    base_images = [image.crop(CONTENT).resize((WIDTH,HEIGHT),Image.Resampling.LANCZOS)
                   for image in read_frames(args.base,data['times'],(256,256))]
    with Image.open(args.reference) as image:
        original = ImageOps.exif_transpose(image).convert('RGB')
        if original.width * HEIGHT != original.height * WIDTH:
            raise ValueError('reference and native teacher aspect ratios differ')
        reference = to_tensor([original.resize((WIDTH,HEIGHT),Image.Resampling.LANCZOS).crop(BOX)])
    teacher = to_tensor([image.crop(BOX) for image in target_images])
    base = to_tensor([image.crop(BOX) for image in base_images])
    # Restrict the edit to a soft halo around the teacher's outer contour.
    outer = guides[:,0:1]
    alpha = F.avg_pool2d(F.max_pool2d(outer,23,1,11),9,1,4,count_include_pad=False)
    if any(torch.any(edge != 0) for edge in (alpha[...,0,:],alpha[...,-1,:],alpha[...,:,0],alpha[...,:,-1])):
        raise ValueError('mouth halo touches crop boundary')
    inner = guides[:,2:3]
    focus = alpha*(1+3*outer+5*inner)
    lip_edges = alpha*(1+3*guides[:,1:2]+4*guides[:,3:4])
    active = np.flatnonzero(mouth[1:,3] >= np.quantile(mouth[1:,3],.75))
    model = LocalRenderer(guides.shape[1])
    optimizer = torch.optim.Adam(model.parameters(),lr=.0008)
    rng = np.random.default_rng(713)
    args.output.mkdir(parents=True,exist_ok=False)
    for step in range(1,args.steps+1):
        batch = np.concatenate((rng.choice(active,2),rng.integers(0,len(base),2)))
        both = np.concatenate((batch,np.maximum(batch-1,0)))
        optimizer.zero_grad(set_to_none=True)
        predicted = model(reference,base[both],guides[both],alpha[both])
        delta = predicted-teacher[both]
        high = delta-F.avg_pool2d(delta,5,1,2,count_include_pad=False)
        loss = (masked_mean(delta.abs(),focus[both])+.75*edge_error(delta,lip_edges[both])
                +.5*masked_mean(high.abs(),focus[both])
                +.20*masked_mean((delta[:4]-delta[4:]).abs(),torch.maximum(alpha[batch],alpha[both[4:]])))
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite mouth detail training loss')
        loss.backward()
        optimizer.step()
        if step in (1,args.steps) or step%100==0:
            print(f'step={step} loss={float(loss.detach()):.6f}',flush=True)
    with torch.inference_mode():
        refined = torch.cat([model(reference,b,g,a) for b,g,a in zip(base.split(8),guides.split(8),alpha.split(8))])
    if not torch.isfinite(refined).all():
        raise RuntimeError('nonfinite mouth detail prediction')
    unchanged = (alpha == 0).expand_as(base)
    if not torch.equal(refined[unchanged],base[unchanged]):
        raise RuntimeError('mouth renderer changed pixels outside the edit mask')
    def metrics(output):
        delta = output-teacher
        return {'lip_rgb_mae':float(masked_mean(delta.abs(),outer)),
                'lip_edge_mae':float(edge_error(delta,outer)),
                'open_mouth_rgb_mae':float(masked_mean(delta[active].abs(),outer[active])),
                'open_mouth_edge_mae':float(edge_error(delta[active],outer[active]))}

    original_frames = np.stack([np.asarray(image) for image in base_images])
    output_frames = original_frames.copy()
    x0,y0,x1,y1 = BOX
    output_frames[:,y0:y1,x0:x1] = to_pixels(refined)
    teacher_frames = np.stack([np.asarray(image) for image in target_images])
    write_video(args.output/'student.mp4',output_frames,data['audio'])
    write_video(args.output/'comparison.mp4',np.concatenate((teacher_frames,original_frames,output_frames),axis=2),data['audio'])
    review_image(args.output/'details.jpg',[('teacher',teacher_frames),('previous',original_frames),('detail',output_frames)])
    torch.save({'schema_version':1,'architecture':'native_lip_detail_probe','box':BOX,
                'state_dict':model.state_dict()},args.output/'checkpoint.pt')
    report = {'dataset_manifest_sha256':sha256(args.dataset),'base_video_sha256':sha256(args.base),
              'raw_track_sha256':sha256(args.tracks),'frame_size':[WIDTH,HEIGHT],'mouth_box':BOX,
              'steps':args.steps,'frames':record['frames'],'open_mouth_frames':len(active),
              'previous':metrics(base),'detail':metrics(refined),'outside_edit_mask_max_change':0,
              'same_clip_diagnostic':True,'teacher_lip_contours_at_inference':True,
              'audio_driven_sync_verified':False,'mobile_verified':False}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','teacher','reference','base','tracks','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--steps',type=int,default=550)
    args=parser.parse_args()
    if not 1 <= args.steps <= 1000:
        parser.error('--steps must be 1..1000')
    run(args)
