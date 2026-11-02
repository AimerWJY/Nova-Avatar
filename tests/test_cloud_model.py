import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    'cloud_model', Path(__file__).resolve().parents[1] / 'scripts/cloud_model.py')
model = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(model)


def tensor_file(path, key='weight'):
    header = json.dumps({key: {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
    path.write_bytes(struct.pack('<Q', len(header)) + header + b'\0' * 4)


class ModelTests(unittest.TestCase):
    def setup_source(self, root):
        root.mkdir()
        (root / model.INDEX).write_text(json.dumps({'weight_map': {'weight': 'one.safetensors'}}))
        for name in model.COMPONENTS:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if name in ('google/umt5-xxl', 'wav2vec2-large-xlsr-53-english'):
                path.mkdir()
            else:
                path.write_bytes(b'test')

    def test_header_and_truncation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'one.safetensors'
            tensor_file(path)
            model.check_header(path, ['weight'])
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, 'payload'):
                model.check_header(path, ['weight'])

    def test_reject_path_traversal(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / model.INDEX).write_text(json.dumps({'weight_map': {'w': '../bad'}}))
            with self.assertRaisesRegex(ValueError, 'unsafe'):
                model.weight_map(root)

    def test_complete_source_repair_is_offline_and_repeatable(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / 'shared', Path(folder) / 'local'
            self.setup_source(source)
            tensor_file(source / 'one.safetensors')
            with patch.dict('sys.modules', {'huggingface_hub': None}):
                model.repair(source, target)
                model.repair(source, target)
            self.assertTrue((target / 'one.safetensors').is_symlink())
            self.assertEqual(model.inspect(target), [])

    def test_missing_shard_downloads_only_once(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / 'shared', Path(folder) / 'local'
            self.setup_source(source)
            calls = []
            def download(repo, filename, revision=None, local_dir=None):
                calls.append(filename)
                if filename == model.INDEX:
                    return str(source / model.INDEX)
                path = Path(local_dir) / filename
                tensor_file(path)
                return str(path)
            hub = SimpleNamespace(HfApi=lambda: SimpleNamespace(
                model_info=lambda repo: SimpleNamespace(sha='fixed-revision')),
                hf_hub_download=download)
            with patch.dict('sys.modules', {'huggingface_hub': hub}):
                model.repair(source, target)
                model.repair(source, target)
            self.assertEqual(calls, [model.INDEX, 'one.safetensors'])
            self.assertFalse((source / 'one.safetensors').exists())
            self.assertFalse((target / 'one.safetensors').is_symlink())

    def test_conflict_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / 'shared', Path(folder) / 'local'
            self.setup_source(source)
            tensor_file(source / 'one.safetensors')
            target.mkdir()
            conflict = target / 'config.json'
            conflict.write_text('keep me')
            with self.assertRaisesRegex(ValueError, 'conflicting'):
                model.repair(source, target)
            self.assertEqual(conflict.read_text(), 'keep me')

    def test_target_in_shared_directory_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'shared'
            self.setup_source(source)
            with self.assertRaisesRegex(ValueError, 'non-nested'):
                model.repair(source, source / 'local')

    def test_remote_index_mismatch_stops_before_shard_download(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / 'shared', Path(folder) / 'local'
            self.setup_source(source)
            remote = Path(folder) / 'remote.json'
            remote.write_text(json.dumps({'weight_map': {'other_weight': 'one.safetensors'}}))
            calls = []
            def download(repo, filename, **kwargs):
                calls.append(filename)
                return str(remote)
            hub = SimpleNamespace(HfApi=lambda: SimpleNamespace(
                model_info=lambda repo: SimpleNamespace(sha='other-revision')),
                hf_hub_download=download)
            with patch.dict('sys.modules', {'huggingface_hub': hub}):
                with self.assertRaisesRegex(ValueError, 'index mismatch'):
                    model.repair(source, target)
            self.assertEqual(calls, [model.INDEX])
            self.assertFalse((target / 'one.safetensors').exists())


if __name__ == '__main__':
    unittest.main()
