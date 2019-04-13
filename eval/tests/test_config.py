import unittest
from pathlib import Path
from eval.config import Config


class TestConfig(unittest.TestCase):

    def test_from_module_path(self):
        mod = 'eval.tests.settings'
        c = Config.from_module_path(mod)
        self.assertEqual(c.dry_run, False)
        self.assertEqual(c.models, ['HRED'])
        self.assertEqual(c.metrics, ['BLEU'])

    def test_ignore_case(self):
        c = Config(dry_run=True)
        c2 = Config(DRY_RUN=True)
        self.assertEqual(c, c2)

    def test_from_filename(self):
        filename = Path(__file__).parent / 'settings.py'
        
