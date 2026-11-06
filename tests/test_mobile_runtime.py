import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    'mobile_runtime', Path(__file__).resolve().parents[1] / 'scripts/mobile_runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class MobileRuntimeTests(unittest.TestCase):
    def test_archive_paths_cannot_escape(self):
        for name in ['/tmp/lib.so', '../lib.so', 'arm64-v8a/../../lib.so', 'a\\b']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                runtime.safe_member(name)
        self.assertEqual(str(runtime.safe_member('release/arm64-v8a/lib/libncnn.so')),
                         'release/arm64-v8a/lib/libncnn.so')

    def test_wrong_archive_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / 'untrusted.zip'
            archive.write_bytes(b'not the official runtime')
            with patch.object(runtime, 'REPO', root), self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                runtime.install(archive)
            self.assertEqual(list((root / 'cache/mobile').iterdir()), [])


if __name__ == '__main__':
    unittest.main()
