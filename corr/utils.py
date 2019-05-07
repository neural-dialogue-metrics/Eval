import json
import logging
import re
from pathlib import Path

import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import scale as sklearn_scale

from eval.consts import SAMPLE_SIZE, RANDOM_STATE, SEPARATOR

logger = logging.getLogger(__name__)

DATA_FILENAME_RE = re.compile(r'\w+-\w+-\w+\.json')


def scale_and_sample(frame: pd.DataFrame):
    return frame.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).transform(sklearn_scale)


class Triple:
    def __init__(self, model, dataset, metric):
        self.model = model
        self.dataset = dataset
        self.metric = metric

    @property
    def parts(self):
        return self.model, self.dataset, self.metric

    @property
    def name(self):
        return SEPARATOR.join((self.model, self.dataset, self.metric))


class UtterScoreDist(Triple):
    """Utterance-Score Distribution"""

    def __init__(self, model, dataset, metric, system, utterance):
        super(UtterScoreDist, self).__init__(model, dataset, metric)
        self.system = system
        self.utterance = utterance

    @classmethod
    def from_json_file(cls, filename):
        data = json.load(open(filename))
        return cls(**data)


def find_all_data_files(dist_dir):
    dist_dir = Path(dist_dir)
    data_files = filter(lambda path: DATA_FILENAME_RE.match(path.name), dist_dir.glob('*.json'))
    return list(data_files)


def is_fully_substituted(url):
    return re.search(r'<[\w_\d]+>', url) is None


def load_filename_data(prefix):
    data_files = find_all_data_files(prefix)

    def parse(path: Path):
        model, dataset, metric = path.stem.split(SEPARATOR)
        return locals()

    return DataFrame.from_records([parse(p) for p in data_files])


def substitute_url(url: str, check_full=False, **kwargs):
    for name, value in kwargs.items():
        url = url.replace('<{}>'.format(name), value)
    if check_full and not is_fully_substituted(url):
        raise ValueError('url not fully substituted: {}'.format(url))
    return url
