"""Diagnostic renderer driven by observed teacher landmarks on one reviewed clip."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.student_train import write_video
from student.data import load, sha256

JOINTS = tuple(f'{side}_{joint}' for side in ('left', 'right') for joint in ('shoulder', 'elbow', 'wrist'))
SAMPLES = (0, 6, 10, 14, 18, 24, 40, 64)
SIZE = 256


def point_xy(point):
    coords = np.array([point['x'], point['y']], dtype=np.float64)
    if not np.isfinite(coords).all() or np.any((coords < 0) | (coords > 1)):
        raise ValueError('landmark is outside the input image')
    return coords * [192, 256] + [32, 0]


def observations(raw, count):
    rows = {item['file']: item for item in raw}
    names = ['reference.png', *(f'{i:03}.png' for i in range(count))]
    if len(raw) != len(names) or set(rows) != set(names):
        raise ValueError('landmarks must cover the reference and every prepared video frame')
    joints, mouth, scores = [], [], []
    for name in names:
        row = rows[name]
        scores.append([row['body'].get(joint, {}).get('confidence', 0) for joint in JOINTS])
        joints.append([point_xy(row['body'][joint]) if joint in row['body'] else [np.nan]*2 for joint in JOINTS])
        inner = np.array([point_xy(point) for point in row['innerLips']])
        outer = np.array([point_xy(point) for point in row['outerLips']])
        if len(inner) < 4 or len(outer) < 4:
            raise ValueError(f'missing lip contour in {name}')
        mouth.append([*outer.mean(axis=0), np.ptp(outer[:, 0]), np.ptp(inner[:, 1])])
    joints = np.asarray(joints)
    scores = np.asarray(scores)
    if not np.isfinite(scores).all() or np.any(scores < .25) or not np.isfinite(joints).all():
        weak = np.argwhere((scores < .25) | ~np.isfinite(joints).all(axis=2))
        raise ValueError(f'low-confidence body landmarks need manual review: {weak.tolist()}')
    return joints, np.asarray(mouth), scores


def radial(distance_squared):
    return distance_squared * np.log(np.maximum(distance_squared, 1e-12))


def warp_grid(source, target):
    # Fixed outer and torso points reduce the influence of arm motion on the face and background.
    anchors = np.array([[32, 0], [224, 0], [32, 128], [224, 128], [32, 255], [224, 255],
                        [128, 65], [128, 108], [128, 150], [128, 195], [128, 240]])
    controls = np.concatenate((target, anchors)) / SIZE
    displacement = np.concatenate((source-target, np.zeros_like(anchors))) / SIZE
    poly = np.column_stack((np.ones(len(controls)), controls))
    diff = controls[:, None] - controls[None, :]
    kernel = radial((diff**2).sum(axis=-1))
    system = np.block([[kernel + np.eye(len(controls))*1e-5, poly], [poly.T, np.zeros((3, 3))]])
    weights = np.linalg.solve(system, np.vstack((displacement, np.zeros((3, 2)))))
    yy, xx = np.mgrid[:SIZE, :SIZE]
    pixels = np.stack((xx.ravel(), yy.ravel()), axis=-1) / SIZE
    distances = ((pixels[:, None] - controls)**2).sum(axis=-1)
    basis = np.column_stack((radial(distances), np.ones(len(pixels)), pixels))
    sampling_pixels = pixels + basis @ weights
    return ((sampling_pixels.reshape(SIZE, SIZE, 2)+.5/SIZE)*2-1).astype(np.float32)


def pose_planes(joints, mouth):
    yy, xx = np.mgrid[:64, :64]*4 + 1.5
    points = np.vstack((joints, mouth[:2]))
    distances = (xx[None]-points[:, 0, None, None])**2 + (yy[None]-points[:, 1, None, None])**2
    width = np.array([8, 8, 8, 8, 8, 8, 5])[:, None, None]
    heat = np.exp(-.5*distances/width**2)
    return np.concatenate((heat, heat[-1:] * (mouth[3]/8), heat[-1:] * (mouth[2]/20))).astype(np.float32)


class PoseResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.low = nn.Sequential(nn.Conv2d(15, 24, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(24, 24, 3, padding=1), nn.SiLU())
        self.middle = nn.Sequential(nn.Conv2d(24, 16, 3, padding=1), nn.SiLU())
        self.high = nn.Sequential(nn.Conv2d(22, 12, 3, padding=1), nn.SiLU(),
                                  nn.Conv2d(12, 3, 3, padding=1))
        nn.init.zeros_(self.high[-1].weight)
        nn.init.zeros_(self.high[-1].bias)

    def forward(self, reference, warped, pose):
        images = torch.cat((reference.expand_as(warped), warped), dim=1)
        reduced = F.avg_pool2d(images, 4)
        value = self.low(torch.cat((reduced, pose), dim=1))
        value = self.middle(F.interpolate(value, scale_factor=2, mode='bilinear', align_corners=False))
        value = F.interpolate(value, scale_factor=2, mode='bilinear', align_corners=False)
        return (warped + self.high(torch.cat((value, images), dim=1)).tanh()).clamp(0, 1)


def draw_tracks(path, images, joints, mouths):
    board = Image.new('RGB', (4*SIZE, 2*(SIZE+22)), '#222222')
    draw = ImageDraw.Draw(board)
    for slot, index in enumerate(SAMPLES):
        image = Image.fromarray(images[index])
        overlay = ImageDraw.Draw(image)
        for offset, color in ((0, 'cyan'), (3, 'orange')):
            arm = [tuple(point) for point in joints[index, offset:offset+3]]
            overlay.line(arm, fill=color, width=2)
            for x,y in arm:
                overlay.ellipse((x-2,y-2,x+2,y+2), fill=color)
        x,y,w,h = mouths[index]
        overlay.rectangle((x-w/2,y-h/2-1,x+w/2,y+h/2+1), outline='magenta', width=1)
        col,row = slot%4,slot//4
        board.paste(image, (col*SIZE,row*(SIZE+22)+22))
        draw.text((col*SIZE+6,row*(SIZE+22)+5), f'{index/16:.3f}s', fill='white')
    board.save(path)


def run(args):
    torch.set_num_threads(2)
    torch.manual_seed(713)
    record, data = load(args.dataset)
    padded = all(not (pixels[..., :32, :].any() or pixels[..., 224:, :].any())
                 for pixels in (data['reference'], data['frames']))
    if record['frames'] <= max(SAMPLES) or not padded:
        raise ValueError('expected the 3:4 wide diagnostic pair with verified side padding')
    raw = json.loads(args.tracks.read_text())
    joints, mouths, scores = observations(raw, record['frames'])
    args.output.mkdir(parents=True, exist_ok=False)
    draw_tracks(args.output/'teacher_tracks.jpg', data['frames'], joints[1:], mouths[1:])
    (args.output/'tracks.json').write_text(json.dumps({
        'fps': 16, 'joint_order': JOINTS, 'coordinate_space': '256x256 pixels, upper-left origin',
        'reference_joints': joints[0].tolist(), 'reference_mouth': mouths[0].tolist(),
        'mouth_fields': ['center_x', 'center_y', 'outer_width', 'inner_height'],
        'joints': joints[1:].tolist(), 'mouth': mouths[1:].tolist(), 'body_confidence': scores[1:].tolist(),
    }, indent=2)+'\n')
    print('Verified tracks; preparing geometry', flush=True)
    reference = torch.from_numpy(data['reference']).permute(2,0,1)[None].float()/255
    target = torch.from_numpy(data['frames']).permute(0,3,1,2).float()/255
    grids = torch.from_numpy(np.stack([warp_grid(joints[0], row) for row in joints[1:]]))
    poses = torch.from_numpy(np.stack([pose_planes(arm, mouth) for arm,mouth in zip(joints[1:],mouths[1:])]))
    warped = F.grid_sample(reference.expand(record['frames'], -1,-1,-1), grids,
                           padding_mode='border', align_corners=False)
    mask = torch.zeros((1,1,SIZE,SIZE)); mask[:,:,:,32:224] = 1
    warped = warped*mask
    weights = torch.ones((record['frames'],1,SIZE,SIZE))
    # These fixed ROIs include the full observed range of both arms and the mouth.
    weights[:,:,165:255,52:106] += 2
    weights[:,:,165:255,150:204] += 2
    weights[:,:,51:77,112:144] += 5
    model = PoseResidual()
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    rng = np.random.default_rng(713)
    movement = np.linalg.norm(joints[1:, [2,5]]-joints[0, [2,5]], axis=-1).max(axis=1)
    active = np.flatnonzero(movement >= np.quantile(movement,.75))
    for step in range(1,args.steps+1):
        batch = np.concatenate((rng.choice(active,2), rng.integers(0,record['frames'],2)))
        opt.zero_grad(set_to_none=True)
        result = model(reference,warped[batch],poses[batch])*mask
        error = (result-target[batch]).abs()
        loss = (error*weights[batch]).mean()
        if not torch.isfinite(loss):
            raise RuntimeError('non-finite pose rendering loss')
        loss.backward()
        opt.step()
        if step==1 or step%50==0 or step==args.steps:
            print(f'step={step} loss={float(loss.detach()):.6f}',flush=True)
    with torch.inference_mode():
        refined = torch.cat([model(reference,w,p)*mask for w,p in zip(warped.split(4),poses.split(4))])
    def pixels(tensor):
        return tensor.permute(0,2,3,1).mul(255).round().byte().numpy()
    for name,result in (('warp',warped),('refined',refined)):
        write_video(args.output/f'{name}.mp4',pixels(result),data['audio'])
    write_video(args.output/'comparison.mp4',np.concatenate((data['frames'],pixels(warped),pixels(refined)),axis=2),data['audio'])
    def metrics(rgb):
        error = (rgb-target).abs()
        return {'full_mae':float(error.mean()),'mouth_roi_mae':float(error[:,:,51:77,112:144].mean()),
                'arm_roi_mae_first_29_frames':float(torch.cat((error[:29,:,165:255,52:106],error[:29,:,165:255,150:204]),dim=-1).mean())}
    torch.save({'schema_version':1,'architecture':'oracle_pose_residual_probe','state_dict':model.state_dict()},args.output/'checkpoint.pt')
    report = {'dataset_manifest_sha256':sha256(args.dataset),'raw_track_sha256':sha256(args.tracks),
              'frames':record['frames'],'steps':args.steps,'conditioning':'teacher landmarks at every frame',
              'minimum_body_confidence':float(scores.min()),'maximum_reference_to_teacher_wrist_shift_px':float(movement.max()),
              'teacher_inner_lip_opening_range_px':[float(mouths[1:,3].min()),float(mouths[1:,3].max())],
              'static_reference':metrics(reference), 'pose_warp':metrics(warped),'pose_residual':metrics(refined),
              'same_clip_diagnostic':True,'audio_to_pose_verified':False,'mobile_verified':False}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--tracks',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=200)
    args = parser.parse_args()
    if not 1 <= args.steps <= 1000:
        parser.error('--steps must be 1..1000')
    run(args)
