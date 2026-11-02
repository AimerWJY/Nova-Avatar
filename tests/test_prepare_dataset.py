import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    'prepare_dataset', Path(__file__).resolve().parents[1] / 'scripts/prepare_dataset.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class DatasetTests(unittest.TestCase):
    def test_missing_audio_rejected(self):
        with patch.object(module, 'probe', return_value={'streams': [{'codec_type': 'video'}]}):
            with self.assertRaisesRegex(ValueError, 'embedded audio'):
                module.inspect(Path('video.mp4'))

    def test_vfr_and_duration_warning(self):
        info = {'streams': [
            {'codec_type': 'video', 'duration': '5', 'width': 832, 'height': 480,
             'avg_frame_rate': '25/1', 'nb_frames': '125'},
            {'codec_type': 'audio', 'duration': '4.5', 'sample_rate': '48000'}]}
        with patch.object(module, 'probe', return_value=info), patch.object(
                module, 'run', return_value='0.0\n0.04\n0.12\n'):
            details, warnings = module.inspect(Path('video.mp4'))
        self.assertTrue(details['variable_frame_timing'])
        self.assertTrue(any('100ms' in warning for warning in warnings))

    def test_scan_deduplicates_and_reports_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            incoming = root / 'incoming'
            incoming.mkdir()
            (incoming / 'one.mp4').write_bytes(b'same')
            (incoming / 'two.mp4').write_bytes(b'same')
            (incoming / 'bad.mp4').write_bytes(b'bad')
            output = root / 'dataset'
            def inspect(path):
                if path.name == 'bad.mp4':
                    raise ValueError('broken input')
                return {'duration_seconds': 5}, []
            argv = ['prepare_dataset.py', 'scan', '--input', str(incoming),
                    '--output', str(output), '--person-id', 'p1',
                    '--source-type', 'real', '--rights-note', 'test permission']
            with patch('sys.argv', argv), patch.object(module.shutil, 'which', return_value='/fake'), \
                    patch.object(module, 'inspect', side_effect=inspect):
                self.assertEqual(module.main(), 1)
            report = json.loads(next((output / 'manifests').glob('report_*.json')).read_text())
            self.assertEqual(report['accepted'], 1)
            self.assertEqual(len(report['duplicates']), 1)
            self.assertEqual(len(report['errors']), 1)
            record = json.loads(next((output / 'manifests').glob('scan_*.jsonl')).read_text())
            self.assertEqual(record['review']['status'], 'pending')
            self.assertEqual(record['split'], 'unassigned')
            self.assertFalse((output / 'originals').exists())

    def test_failed_asset_does_not_leave_partial_file(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'audio.wav'
            def fail(path):
                path.write_bytes(b'partial')
                raise ValueError('conversion failed')
            with self.assertRaises(ValueError):
                module.atomic_asset(target, fail)
            self.assertEqual(list(Path(folder).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
