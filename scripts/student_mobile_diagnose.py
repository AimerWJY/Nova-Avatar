"""Compare dumped v2 nodes with PyTorch, separating upstream inputs from local kernel error."""
import argparse
import json
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def read_nodes(directory):
    import numpy as np
    rows = json.loads((directory/'nodes.json').read_text()); result = {}
    for row in rows:
        if (not re.fullmatch(r'node_\d+\.f32', row['file']) or len(row['shape']) != 4
                or any(type(value) is not int or value <= 0 for value in row['shape'])):
            raise ValueError('invalid diagnostic tensor metadata')
        count = math.prod(row['shape']); path = directory/row['file']
        if count*4 > 64*1024*1024 or path.is_symlink() or path.stat().st_size != count*4:
            raise ValueError('invalid diagnostic tensor size/path')
        values = np.fromfile(path, dtype='<f4')
        if not np.isfinite(values).all():
            raise ValueError('non-finite diagnostic tensor')
        result[row['layer']] = {**row, 'values': values}
    return result


def diagnose(phone_report, model_manifest, checkpoint_path, dataset, output):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from student.data import load, sha256
    from student.audio import histories
    from student.mouth_model import MouthCandidate

    torch.set_num_threads(2)
    report = json.loads(phone_report.read_text()); export = json.loads(model_manifest.read_text())
    if (report['model_manifest_sha256'] != sha256(model_manifest)
            or export['checkpoint_sha256'] != sha256(checkpoint_path)
            or export['dataset_manifest_sha256'] != sha256(dataset)):
        raise ValueError('diagnostic sources do not match tested artifacts')
    frame_index = report['native']['diagnostic_frame']
    if frame_index < 0:
        raise ValueError('phone report did not dump a diagnostic frame')
    nodes = read_nodes(phone_report.parent/'diagnostics')
    model = MouthCandidate(); model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True)['state_dict']); model.eval()
    def tensor(name, shape):
        return torch.from_numpy(nodes[name]['values'].copy()).reshape(shape)
    native_inputs = [tensor('in0',(1,3,256,256)), tensor('in1',(1,32,64,64)), tensor('in2',(1,128)),
                     tensor('in3',(1,8,40)), tensor('in4',(1,128))]
    mapping = {'audio':'gemm_0', 'attention.q':'gemm_1', 'attention.k':'gemm_2', 'attention.v':'gemm_3',
               'attention.out':'gemm_4', 'audio_context':'linear_37', 'motion':'linear_38', 'modulate':'linear_39',
               'decoder1':'conv_29', 'decoder2':'conv_30', 'output':'conv_31',
               'mouth.encoder.0':'conv_32', 'mouth.encoder.2':'conv_33', 'mouth.condition.0':'linear_40',
               'mouth.condition.2':'linear_41', 'mouth.fuse.0':'conv_34', 'mouth.refine.0':'conv_35', 'mouth.output':'conv_36'}
    def capture(inputs):
        image,spatial,identity,audio,state = inputs; captured = {}; handles = []
        for name, node in mapping.items():
            handles.append(model.get_submodule(name).register_forward_hook(
                lambda module, args, value, key=node: captured.__setitem__(key, value.detach().clone())))
        handles.append(model.mouth.register_forward_pre_hook(
            lambda module, args: captured.__setitem__('clamp_50', args[0].detach().clone())))
        with torch.inference_mode():
            rgb, next_state = model.forward_frame(audio,state,(spatial,image),identity)
            captured['clamp_51']=rgb; captured['tanh_17']=next_state
            captured['weighted_sum_0']=model.identity_grid+captured['conv_31'][:,:2].tanh().permute(0,2,3,1)*.1875
            captured['gridsample_5']=F.grid_sample(image,captured['weighted_sum_0'],mode='bilinear',padding_mode='border',align_corners=False)
        for handle in handles:
            handle.remove()
        return captured
    identical_inputs = capture(native_inputs)
    _, data = load(dataset)
    image = torch.from_numpy(data['reference'].copy()).permute(2,0,1)[None].float()/255
    audio = torch.from_numpy(histories(data['audio_features']))
    with torch.inference_mode():
        cache,identity=model.encode_identity(image); state=torch.zeros(1,128)
        for row in audio[:frame_index]:
            _,state=model.forward_frame(row[None],state,cache,identity)
    original_inputs = [image,cache[0],identity,audio[frame_index:frame_index+1],state]
    original = capture(original_inputs)
    def error(actual, expected):
        a=np.asarray(actual,dtype=np.float64).reshape(-1); e=np.asarray(expected,dtype=np.float64).reshape(-1)
        if a.shape != e.shape:
            raise ValueError('diagnostic flat size mismatch')
        differences=np.abs(a-e)
        return {'max_abs':float(differences.max()), 'max_scaled':float((differences/(1e-5+1e-4*np.abs(e))).max())}
    rows=[]
    for name,expected in identical_inputs.items():
        if name in nodes:
            rows.append({'layer':name,'type':nodes[name]['type'],
                         'same_native_inputs':error(nodes[name]['values'],expected.numpy()),
                         'original_pipeline':error(nodes[name]['values'],original[name].numpy())})
    sampling_same_grid = None
    if 'weighted_sum_0' in nodes and 'gridsample_5' in nodes:
        grid=tensor('weighted_sum_0',(1,256,256,2))
        with torch.inference_mode():
            warped=F.grid_sample(native_inputs[0],grid,mode='bilinear',padding_mode='border',align_corners=False)
        sampling_same_grid=error(nodes['gridsample_5']['values'],warped.numpy())
    result={'schema_version':1,'phone_report':str(phone_report.resolve()),'frame':frame_index,
            'device_request':report['device_request'], 'sampling_with_identical_grid':sampling_same_grid,
            'input_errors':{name:error(native.numpy(),source.numpy()) for name,native,source in zip(
                ('image','spatial','identity','audio','incoming_state'),native_inputs,original_inputs)},
            'nodes':rows,'validation_threshold_changed':False}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('phone-report','model-manifest','checkpoint','dataset','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();diagnose(a.phone_report,a.model_manifest,a.checkpoint,a.dataset,a.output)


if __name__=='__main__': main()
