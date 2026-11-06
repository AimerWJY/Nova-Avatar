import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('mobile_profile',
    Path(__file__).resolve().parents[1] / 'scripts/mobile_profile.py')
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


class MobileProfileTests(unittest.TestCase):
    def test_memavailable_is_not_memfree(self):
        result = profile.memory_info('MemTotal: 11285460 kB\nMemFree: 265196 kB\nMemAvailable: 5245004 kB\n')
        self.assertEqual(result['MemAvailable'], 5245004 * 1024)
        self.assertGreater(result['MemAvailable'], result['MemFree'])

    def test_missing_available_fails(self):
        with self.assertRaises(ValueError):
            profile.memory_info('MemTotal: 100000 kB\nMemFree: 100 kB\n')

    def test_select_only_authorized_device(self):
        items = profile.devices('List of devices attached\nc9fb5e39\tdevice\nother\tunauthorized\n')
        self.assertEqual(profile.choose_device(items), 'c9fb5e39')
        with self.assertRaises(ValueError):
            profile.choose_device(items, 'other')

    def test_multiple_devices_require_selection(self):
        items = [{'serial': 'a', 'state': 'device'}, {'serial': 'b', 'state': 'device'}]
        with self.assertRaises(ValueError):
            profile.choose_device(items)
        self.assertEqual(profile.choose_device(items, 'b'), 'b')


if __name__ == '__main__':
    unittest.main()
