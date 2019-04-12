import unittest

from eval.model import TwitterSerbanModelFinder, UbuntuSerbanModelFinder
from eval.model import ModelInfo, ModelOutput

TWITTER_ROOT = '/home/cgsdfc/TwitterDialogueCorpus'
UBUNTU_ROOT = '/home/cgsdfc/UbuntuDialogueCorpus'


class TestFindSerbanModels(unittest.TestCase):
    def _check_output(self, models):
        for m in models:
            output: ModelOutput = m.output
            self.assertTrue(output.response_path.is_file())
            
            # NBest is optional.
            if output.nbest_path:
                self.assertTrue(output.nbest_path.is_file())

    def test_UbuntuFinder(self):
        finder = UbuntuSerbanModelFinder(UBUNTU_ROOT)
        models = list(finder.find_models())
        self._check_output(models)

    def test_TwitterFinder(self):
        finder = TwitterSerbanModelFinder(TWITTER_ROOT)
        models = list(finder.find_models())
        self._check_output(models)
