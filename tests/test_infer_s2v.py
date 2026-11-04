import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
spec = importlib.util.spec_from_file_location('cloud_model', SCRIPTS / 'cloud_model.py')
cloud_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cloud_model)
spec = importlib.util.spec_from_file_location('infer_s2v', SCRIPTS / 'infer_s2v.py')
inference = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {'cloud_model': cloud_model}):
    spec.loader.exec_module(inference)


class InferenceTests(unittest.TestCase):
    def settings(self, **overrides):
        values = dict(width=480, height=640, seconds=5, steps=40, vram_limit=20)
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_frame_alignment_and_no_audio_overrun(self):
        for duration in (1, 1.01, 2.6, 4.99, 5, 5.8, 30):
            with self.subTest(duration=duration):
                frames = inference.frame_count(duration, 5)
                self.assertEqual(frames % 4, 1)
                self.assertLessEqual((frames - 1) / 16, min(duration, 5) + 1e-8)
        self.assertEqual(inference.frame_count(6, 5), 81)
        self.assertEqual(inference.frame_count(4.99, 5), 77)

    def test_invalid_settings_blocked(self):
        for changes in ({'width': 481}, {'height': 0}, {'seconds': 30},
                        {'steps': 0}, {'seconds': float('nan')}, {'vram_limit': 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                inference.validate_settings(self.settings(**changes))

    def test_pipeline_uses_local_paths_and_limits(self):
        assets = {key: f'/shared/{key}' for key in
                  ('audio_encoder', 'text_encoder', 'vae', 'tokenizer', 'audio_processor')}
        assets['dit'] = ['/shared/shard1', '/shared/shard2']
        plan = {'assets': assets, 'offload': 'disk', 'effective_vram_limit_gib': 20}
        config = Mock(side_effect=lambda **kwargs: kwargs)
        pipeline = Mock()
        inference.build_pipeline(plan, SimpleNamespace(bfloat16='bf16'), pipeline, config)
        kwargs = pipeline.from_pretrained.call_args.kwargs
        self.assertFalse(kwargs['redirect_common_files'])
        self.assertEqual(kwargs['vram_limit'], 20)
        self.assertEqual(kwargs['model_configs'][0]['path'], assets['dit'])
        for model in kwargs['model_configs']:
            self.assertEqual(model['offload_device'], 'disk')
            self.assertEqual(model['computation_device'], 'cuda')
            self.assertNotIn('model_id', model)
        plan['offload'] = 'cpu'
        inference.build_pipeline(plan, SimpleNamespace(bfloat16='bf16'), pipeline, config)
        self.assertEqual(pipeline.from_pretrained.call_args.kwargs['model_configs'][0]['offload_device'], 'cpu')

    def test_backend_revision_mismatch_rejected(self):
        distribution = SimpleNamespace(version='x', read_text=lambda name: json.dumps(
            {'vcs_info': {'commit_id': 'wrong'}}))
        with patch.object(inference.importlib.metadata, 'distribution', return_value=distribution):
            with self.assertRaisesRegex(ValueError, 'revision'):
                inference.backend_provenance()

    def test_failed_inference_preserves_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = ['infer_s2v.py', '--root', str(root), '--model-dir', str(root / 'shared')]
            with patch('sys.argv', args), patch.object(inference, 'make_plan', return_value={'test': True}), \
                    patch.object(inference, 'execute', side_effect=RuntimeError('simulated OOM')), \
                    patch.object(inference.traceback, 'print_exc'):
                self.assertEqual(inference.main(), 1)
            report = json.loads(next((root / 'outputs').glob('*/report.json')).read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertIn('simulated OOM', report['error'])
            self.assertIn('total_after_preflight', report['timings_seconds'])

    def test_dry_run_does_not_execute_or_create_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = ['infer_s2v.py', '--root', str(root), '--dry-run']
            with patch('sys.argv', args), patch.object(inference, 'make_plan', return_value={'test': True}), \
                    patch.object(inference, 'execute') as execute:
                self.assertEqual(inference.main(), 0)
                execute.assert_not_called()
            self.assertFalse((root / 'outputs').exists())


if __name__ == '__main__':
    unittest.main()
