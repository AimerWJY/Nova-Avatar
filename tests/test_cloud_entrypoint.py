import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/cloud.sh'


class EntrypointTests(unittest.TestCase):
    def check_arguments(self, override=None):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            activate = root / '.venv/bin/activate'
            activate.parent.mkdir(parents=True)
            # Capture dispatch arguments without requiring a GPU or real venv.
            activate.write_text('python() { printf "%s\\n" "$@"; }\n')
            env = dict(os.environ, NOVA_ROOT=str(root), NOVA_SHARED_MODEL=str(root / 'shared'))
            env.pop('NOVA_VENV', None)
            env.pop('NOVA_MODEL_DIR', None)
            if override:
                env['NOVA_MODEL_DIR'] = override
            result = subprocess.run(['bash', str(SCRIPT), 'model-check'],
                                    env=env, capture_output=True, text=True, check=True)
            arguments = result.stdout.splitlines()
            expected = override or str(root / 'shared')
            self.assertEqual(arguments[1:], ['check', '--model-dir', expected])
            self.assertNotIn('--target', arguments)
            self.assertFalse((root / 'models').exists())

    def test_shared_is_default(self):
        self.check_arguments()

    def test_explicit_override(self):
        self.check_arguments('/custom/model')

    def test_student_export_forwards_arguments(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            activate = root / '.venv/bin/activate'
            activate.parent.mkdir(parents=True)
            activate.write_text('python() { printf "%s\\n" "$@"; }\n')
            env = dict(os.environ, NOVA_VENV=str(root / '.venv'))
            result = subprocess.run(['bash', str(SCRIPT), 'student-export', '--output', '/exports/v0', '--smoke-train'],
                                    env=env, capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.splitlines(), [str(SCRIPT.parent / 'student_export.py'),
                                                         '--output', '/exports/v0', '--smoke-train'])

    def test_infer_forwards_arguments_and_offline_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            activate = root / '.venv/bin/activate'
            activate.parent.mkdir(parents=True)
            activate.write_text('python() { printf "%s\\n" "$@"; '
                                'printf "offline=%s\\n" "$HF_HUB_OFFLINE"; }\n')
            env = dict(os.environ, NOVA_ROOT=str(root), NOVA_VENV=str(root / '.venv'),
                       NOVA_MODEL_DIR='/shared/model')
            result = subprocess.run(['bash', str(SCRIPT), 'infer', '--dry-run', '--seconds', '2'],
                                    env=env, capture_output=True, text=True, check=True)
            self.assertIn('--model-dir\n/shared/model\n--dry-run\n--seconds\n2', result.stdout)
            self.assertIn('offline=1', result.stdout)
            self.assertEqual(len(list((root / 'reports').glob('infer-*.log'))), 1)


if __name__ == '__main__':
    unittest.main()
