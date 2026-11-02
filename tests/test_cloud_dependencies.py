import importlib.util
import subprocess
import unittest
from pathlib import Path
from unittest.mock import Mock

SPEC = importlib.util.spec_from_file_location(
    'cloud_dependencies', Path(__file__).resolve().parents[1] / 'scripts/cloud_dependencies.py')
dependencies = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dependencies)


class DependencyTests(unittest.TestCase):
    def evaluate(self, message=None, wheel=None, version='0.6.0', system='Linux', smoke=None):
        result = subprocess.CompletedProcess([], 1,
            stdout=message if message is not None else dependencies.KNOWN_MESSAGE + '\n', stderr='')
        return dependencies.classify(result, version,
            wheel if wheel is not None else 'Tag: ' + dependencies.KNOWN_TAG,
            system, 'x86_64', smoke or Mock(return_value={'decode': 'pass'}))

    def test_known_tag_requires_decode_and_keeps_warning(self):
        smoke = Mock(return_value={'decode': 'pass'})
        report = self.evaluate(smoke=smoke)
        smoke.assert_called_once()
        self.assertEqual(report['errors'], [])
        self.assertEqual(report['returncode'], 1)
        self.assertTrue(report['warnings'])

    def test_decode_failure_blocks(self):
        report = self.evaluate(smoke=Mock(side_effect=RuntimeError('cannot decode')))
        self.assertTrue(report['errors'])
        self.assertFalse(report['warnings'])

    def test_other_dependency_failure_not_ignored(self):
        smoke = Mock()
        report = self.evaluate(message=dependencies.KNOWN_MESSAGE + '\nother package requires missing\n', smoke=smoke)
        self.assertTrue(report['errors'])
        smoke.assert_not_called()

    def test_other_metadata_platform_or_version_not_ignored(self):
        for kwargs in ({'wheel': 'Tag: cp312-cp312-linux_x86_64'},
                       {'system': 'Darwin'}, {'version': '0.7.0'}):
            with self.subTest(kwargs=kwargs):
                self.assertTrue(self.evaluate(**kwargs)['errors'])

    def test_clean_pip_needs_no_workaround(self):
        smoke = Mock()
        report = dependencies.classify(subprocess.CompletedProcess([], 0, 'No broken requirements found.\n', ''),
                                       None, '', 'Linux', 'x86_64', smoke)
        self.assertEqual(report['errors'], [])
        smoke.assert_not_called()


if __name__ == '__main__':
    unittest.main()
