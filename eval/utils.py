from typing import Sequence

import logging
import numpy as np
from pathlib import Path

from eval.consts import *

logger = logging.getLogger(__name__)


def ruber_data(train_dir, data_dir, embedding):
    data_dir = Path(data_dir)
    query_vocab = data_dir.glob('*_contexts.*.vocab*')
    query_embed = data_dir.glob('*_contexts.*.embed')
    reply_vocab = data_dir.glob('*_responses.*.vocab*')
    reply_embed = data_dir.glob('*_responses.*.embed')
    return {
        'train_dir': train_dir,
        'query_vocab': query_vocab,
        'query_embed': query_embed,
        'reply_vocab': reply_vocab,
        'reply_embed': reply_embed,
        'embedding': embedding,
    }


class Dataset:
    def __init__(self, name, contexts, references,
                 test_dialogues=None, vocabulary=None, train_set=None):
        self.name = name
        self.contexts = contexts
        self.references = references
        self.test_dialogues = test_dialogues
        self.vocabulary = vocabulary
        self.train_set = train_set

    def __repr__(self):
        return '<{} {}>'.format(self.__class__.__qualname__, self.name)


class UnderTest:

    def __init__(self, metric, model, dataset):
        if model.trained_on != dataset.name:
            raise ValueError('model {} was not trained on dataset {}'.format(model.name, dataset.name))
        self.metric = metric
        self.model = model
        self.dataset = dataset

    def __repr__(self):
        return f'<{self.__class__.__qualname__}: {self.model_name}, {self.dataset_name}, {self.metric_name}>'

    @property
    def model_name(self):
        return self.model.name

    @property
    def dataset_name(self):
        return self.dataset.name

    @property
    def metric_name(self):
        return self.metric.fullname

    @property
    def parts(self):
        return self.model_name, self.dataset_name, self.metric_name

    @property
    def prefix(self):
        parts = self.parts
        if any(SEPARATOR in part for part in parts):
            raise ValueError('{!r} is not allowed in names'.format(SEPARATOR))
        return SEPARATOR.join(parts)

    def get_resource_file(self, key):
        return eval('self.{}'.format(key))


def load_template(name):
    filename = Path(__file__).with_name('template').joinpath(name).with_suffix('.sh')
    return filename.read_text()


def subdirs(path: Path):
    for file in path.iterdir():
        if file.is_dir():
            yield file


def get_random_gpu(low=GPU_LOW, high=GPU_HIGH):
    return np.random.randint(low=low, high=high)


def should_make(target: Path, sources: Sequence[Path]):
    if not target.exists():
        return True
    return all(
        src.exists() and src.stat().st_mtime > target.stat().st_mtime
        for src in sources
    )


def make_parent_dirs(filename: Path):
    """
    Make the parent dirs for a given path.
    :param filename:
    :return:
    """
    parent: Path = filename.parent
    parent.mkdir(parents=True, exist_ok=True)
    return filename
