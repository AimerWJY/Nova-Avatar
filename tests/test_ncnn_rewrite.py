import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.ncnn_v2_rewrite import split_attention_gemm


class AttentionRewriteTests(unittest.TestCase):
    def test_generated_head_graph_matches_independent_batched_attention(self):
        source=('7767517\n6 6\nInput q 0 1 q\nInput kt 0 1 kt\n'
                'MatMul qk 2 1 q kt score\nSoftmax probability 1 1 score probs 0=2 1=1\n'
                'Input v 0 1 v\nMatMul av 2 1 probs v context\n')
        rng=np.random.default_rng(1)
        values={'q':rng.normal(size=(4,8,32)).astype(np.float32),
                'kt':rng.normal(size=(4,32,8)).astype(np.float32),
                'v':rng.normal(size=(4,8,32)).astype(np.float32)}
        score=values['q']@values['kt']; probability=np.exp(score-score.max(-1,keepdims=True))
        probability/=probability.sum(-1,keepdims=True); expected=probability@values['v']
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'frame.param';path.write_text(source)
            self.assertEqual(len(split_attention_gemm(path)),2)
            lines=path.read_text().splitlines();self.assertEqual(lines[1],'42 54')
            for line in lines[2:]:
                fields=line.split();kind=fields[0];n,m=map(int,fields[2:4]);ins=fields[4:4+n];outs=fields[4+n:4+n+m]
                params=dict(value.split('=',1) for value in fields[4+n+m:])
                if kind=='Input':continue
                if kind=='Slice':
                    for name,value in zip(outs,np.split(values[ins[0]],4,axis=0)):values[name]=value
                elif kind=='Reshape':
                    shape=([int(params['2'])] if '2' in params else [])+[int(params['1']),int(params['0'])]
                    values[outs[0]]=values[ins[0]].reshape(shape)
                elif kind=='Gemm':values[outs[0]]=values[ins[0]]@values[ins[1]]
                elif kind=='Concat':values[outs[0]]=np.concatenate([values[name] for name in ins],axis=0)
                elif kind=='Softmax':
                    data=values[ins[0]];data=np.exp(data-data.max(-1,keepdims=True));values[outs[0]]=data/data.sum(-1,keepdims=True)
                else:self.fail('unexpected generated operator '+kind)
            np.testing.assert_allclose(values['context'],expected,rtol=1e-6,atol=1e-6)
            with self.assertRaisesRegex(ValueError,'fixed v2 pair'):
                split_attention_gemm(path)

    def test_unknown_matmul_pattern_is_not_modified(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'frame.param';source='7767517\n1 1\nMatMul other 2 1 a b out 0=1\n';path.write_text(source)
            with self.assertRaises(ValueError):split_attention_gemm(path)
            self.assertEqual(path.read_text(),source)
