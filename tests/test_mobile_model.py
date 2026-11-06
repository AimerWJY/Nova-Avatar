import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import mobile_model


class ConvertedModelTests(unittest.TestCase):
    def test_cpu_fallback_and_numerical_failure_are_explicit(self):
        native = {'status': 'numerical_validation_passed', 'all_outputs_finite': True,
                  'max_scaled_error': .5, 'state_reset_max_scaled_error': 0,
                  'vulkan_requested': True, 'compute_layers_without_vulkan_support': 1, 'p95_ms': 40}
        report = mobile_model.summarize(native)
        self.assertEqual(report['status'], 'validated_with_cpu_fallback')
        self.assertFalse(report['gpu_only_execution_verified'])
        self.assertFalse(report['end_to_end_first_playback_verified'])
        self.assertFalse(report['video_quality_validated'])
        native['diagnostic_frame']=5
        self.assertFalse(mobile_model.summarize(native)['warm_compute_16fps_budget_met'])
        native['max_scaled_error'] = 2
        report = mobile_model.summarize(native)
        self.assertFalse(report['phone_model_numerics_validated'])
        self.assertTrue(report['phone_model_run_completed'])
        self.assertFalse(report['warm_compute_16fps_budget_met'])

    def test_graph_and_fixture_integrity_are_checked_before_adb(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); graphs = []
            for name in ('encoder', 'frame'):
                row = {'name': name}
                for extension in ('param', 'bin'):
                    path = root/f'{name}.ncnn.{extension}'; path.write_bytes(b'unit-test artifact')
                    row[extension+'_sha256'] = mobile_model.digest(path)
                graphs.append(row)
            model = {'architecture': 'mouth_refinement_v2', 'checkpoint_sha256': 'a', 'dataset_manifest_sha256': 'b',
                     'ncnn_conversion': {'passed': True, 'graphs': graphs}}
            manifest = root/'model.json'; manifest.write_text(json.dumps(model))
            fixture_file = root/'v2.fixture'; fixture_file.write_bytes(b'fixture')
            fixture = {'schema_version': 1, 'format': 'converted_v2_fixture_v1',
                       'model_manifest_sha256': mobile_model.digest(manifest),
                       'checkpoint_sha256': 'a', 'dataset_manifest_sha256': 'b', 'frames': 8,
                       'fixture_file': 'v2.fixture', 'fixture_sha256': mobile_model.digest(fixture_file), 'fixture_bytes': 7}
            fixture_manifest = root/'fixture.json'; fixture_manifest.write_text(json.dumps(fixture))
            self.assertEqual(len(mobile_model.assets(manifest, fixture_manifest)[2]), 5)
            (root/'frame.ncnn.bin').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                mobile_model.assets(manifest, fixture_manifest)
