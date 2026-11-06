import importlib.util
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

scripts = Path(__file__).resolve().parents[1] / 'scripts'
spec = importlib.util.spec_from_file_location('mobile_profile', scripts / 'mobile_profile.py')
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)
spec = importlib.util.spec_from_file_location('mobile_benchmark', scripts / 'mobile_benchmark.py')
benchmark = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {'mobile_profile': profile}):
    spec.loader.exec_module(benchmark)


class MobileBenchmarkTests(unittest.TestCase):
    def test_cpu_only_is_not_hardware_success(self):
        result = benchmark.summarize({'devices': [
            {'name': 'reference', 'hardware_driver': False, 'results': [{'status': 'success', 'p50_ms': 5}]}]})
        self.assertEqual(result['status'], 'no_valid_hardware_results')
        self.assertFalse(result['can_infer_video_generation_speed'])

        self.assertEqual(len(result['cpu_results']), 1)
        self.assertEqual(result['hardware_results'], [])

    def test_unsupported_or_inaccurate_is_not_a_latency_result(self):
        result = benchmark.summarize({'devices': [
            {'name': 'driver', 'hardware_driver': True, 'results': [
                {'status': 'unsupported'}, {'status': 'numerical_check_failed', 'p50_ms': 1}]}]})
        self.assertEqual(result['status'], 'no_valid_hardware_results')
        self.assertEqual(len(result['unsupported_results']), 1)
        self.assertEqual(len(result['failed_results']), 1)

    def test_hardware_and_cpu_are_separate(self):
        result = benchmark.summarize({'devices': [
            {'name': 'gpu', 'hardware_driver': True, 'results': [{'status': 'success', 'p50_ms': 1}]},
            {'name': 'cpu', 'hardware_driver': False, 'results': [{'status': 'success', 'p50_ms': 10}]}]})
        self.assertEqual(result['status'], 'hardware_microbench_complete')
        self.assertEqual(result['hardware_results'][0]['device'], 'gpu')
        self.assertEqual(result['cpu_results'][0]['device'], 'cpu')

    def test_partial_success_is_not_full_success(self):
        result = benchmark.summarize({'devices': [
            {'name': 'gpu', 'hardware_driver': True, 'results': [
                {'status': 'success', 'p50_ms': 1}, {'status': 'numerical_check_failed'}]}]})
        self.assertEqual(result['status'], 'hardware_microbench_partial')
        self.assertEqual(len(result['hardware_results']), 1)
        self.assertEqual(len(result['failed_results']), 1)

    def test_empty_hardware_device_is_not_success(self):
        result = benchmark.summarize({'devices': [
            {'name': 'gpu', 'hardware_driver': True, 'results': []}]})
        self.assertEqual(result['status'], 'no_valid_hardware_results')

    def test_candidate_throughput_does_not_imply_cold_start_or_video_quality(self):
        native = {'trained': False, 'synthetic_inputs': True, 'devices': [
            {'name': 'gpu', 'hardware_driver': True, 'results': [{
                'status': 'success', 'all_frames_finite': True, 'cpu_reference_frames': 8,
                'max_scaled_error': .1, 'state_reset_max_scaled_error': 0,
                'target_fps': 16, 'p95_ms': 20, 'stream_wall_ms': 4000,
                'frames': 160, 'cold_first_rgb_ms': 5000}]}]}
        result = benchmark.summarize_candidate(native)
        self.assertEqual(result['status'], 'candidate_graph_validated')
        self.assertTrue(result['synthetic_graph_warm_budget_met'])
        self.assertFalse(result['synthetic_graph_cold_2s_budget_met'])
        self.assertFalse(result['video_quality_validated'])
        self.assertFalse(result['can_infer_video_generation_speed'])
        native['trained'] = True
        result = benchmark.summarize_candidate(native)
        self.assertEqual(result['status'], 'candidate_graph_validated')
        self.assertFalse(result['video_quality_validated'])

    def test_candidate_rejects_insufficient_or_failed_reference_validation(self):
        for checked, error in [(0, .1), (8, 2)]:
            native = {'trained': False, 'synthetic_inputs': True, 'devices': [
                {'name': 'gpu', 'hardware_driver': True, 'results': [{
                    'status': 'success', 'all_frames_finite': True,
                    'cpu_reference_frames': checked, 'max_scaled_error': error,
                    'state_reset_max_scaled_error': 0}]}]}
            result = benchmark.summarize_candidate(native)
            self.assertEqual(result['status'], 'candidate_graph_failed')
            self.assertFalse(result['synthetic_graph_warm_budget_met'])
            self.assertFalse(result['synthetic_graph_cold_2s_budget_met'])

    def test_candidate_rejects_failed_external_pytorch_reference(self):
        native = {'trained': False, 'synthetic_inputs': True,
                  'external_pytorch_reference': {'available': True, 'passed': False},
                  'devices': [{'name': 'gpu', 'hardware_driver': True, 'results': [{
                      'status': 'success', 'all_frames_finite': True, 'cpu_reference_frames': 8,
                      'max_scaled_error': .1, 'state_reset_max_scaled_error': 0}]}]}
        result = benchmark.summarize_candidate(native)
        self.assertEqual(result['status'], 'candidate_graph_failed')
        self.assertFalse(result['pytorch_equivalence_validated'])

    def test_export_integrity_and_path_are_checked_before_adb(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bundle = root / 'candidate.bundle'
            bundle.write_bytes(b'fixture')
            record = {'schema_version': 1, 'format': 'nova_candidate_fixture_v1',
                      'architecture': 'recurrent_conditioned_rgb_v0', 'bundle_file': 'candidate.bundle',
                      'frames': 8, 'reference_frames': 8, 'bundle_bytes': 7,
                      'bundle_sha256': hashlib.sha256(bundle.read_bytes()).hexdigest()}
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps(record))
            self.assertEqual(benchmark.load_candidate_export(manifest)[1], bundle.resolve())
            bundle.write_bytes(b'altered')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                benchmark.load_candidate_export(manifest)
            record['bundle_file'] = '../candidate.bundle'
            manifest.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'unsupported'):
                benchmark.load_candidate_export(manifest)


if __name__ == '__main__':
    unittest.main()
