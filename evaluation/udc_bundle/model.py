"""
This module handles the discovery of models and their generated responses.
"""
import pathlib
import collections

from evaluation.udc_bundle import MODEL_ROOT
from evaluation.udc_bundle import FIRST_RESPONSE_SUFFIX
from evaluation.udc_bundle import KNOWN_MODELS

import logging

_logger = logging.getLogger(__file__)


class ModelInfo(collections.namedtuple('ModelInfo', ['name', 'responses', 'first_response'])):
    @property
    def root_dir(self):
        return pathlib.Path(MODEL_ROOT) / self.name

    @property
    def responses_path(self):
        return self.root_dir / self.responses

    @property
    def first_response_path(self):
        return self.root_dir / self.first_response

    @property
    def responses_key(self):
        return '%s-%s' % (self.name, self.responses)

    @property
    def first_response_key(self):
        return '%s-%s' % (self.name, self.first_response)

    def __str__(self):
        return self.name

    def __repr__(self):
        return '<Model %s>' % self.name


def _make_model(model_dir: pathlib.Path):
    """
    Create a ModelInfo from a model_dir.

    :param model_dir:
    :return:
    """
    assert model_dir.is_dir()
    responses, first_response = None, None
    txt_files = list(model_dir.glob('*.txt'))
    assert len(txt_files) == 2

    for txt in txt_files:
        _logger.info('Found model responses file: %s', txt)
        if txt.name.endswith(FIRST_RESPONSE_SUFFIX):
            first_response = txt.name
        else:
            responses = txt.name

    assert responses and first_response
    model = ModelInfo(model_dir.name, responses, first_response)
    _logger.info('Created ModelInfo %r', model)
    return model


def discover_models():
    """
    Discover models under MODEL_ROOT.

    :return: List[ModelInfo].
    """
    if not MODEL_ROOT.is_dir():
        _logger.error('MODEL_ROOT is not a dir!')
        return
    # Find all directory
    model_dirs = [d for d in MODEL_ROOT.iterdir() if d.is_dir() and d.name in KNOWN_MODELS]
    _logger.info('Found %d models', len(model_dirs))

    for d in model_dirs:
        _logger.info('Found model_dir: %s', d)
    return list(map(_make_model, model_dirs))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    discover_models()
