import itertools
import json
import re
from pathlib import Path

import pandas as pd
import seaborn as sns
import logging
import collections

from sklearn.preprocessing import scale as sklearn_scale
from pandas.plotting import scatter_matrix
from eval.repo import get_model, get_dataset
from eval.consts import SAMPLE_SIZE, RANDOM_STATE, CONFIG_JSON, SEPARATOR
from eval.consts import PEARSON

logger = logging.getLogger(__name__)


def scale_and_sample(frame: pd.DataFrame):
    return frame.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).transform(sklearn_scale)


class UtterScoreDist:
    """Utterance-Score Distribution"""

    def __init__(self, model, dataset, metric, system, utterance):
        self.model = model
        self.dataset = dataset
        self.metric = metric
        self.system = system
        self.utterance = utterance

    def get_model(self):
        return get_model(self.model, self.dataset)

    def get_dataset(self):
        return get_dataset(self.dataset)

    def __repr__(self):
        return f'<{self.__class__.__name__}: {self.name}>'

    @property
    def name(self):
        return SEPARATOR.join((self.model, self.dataset, self.metric))

    @classmethod
    def from_json_file(cls, filename):
        data = json.load(open(filename))
        return cls(**data)

    def plot_dist(self):
        data = scale_and_sample(pd.Series(self.utterance))
        return sns.distplot(data)


DATA_FILENAME_RE = re.compile(r'\w+-\w+-\w+\.json')


def load_dists_from_dir(dist_dir):
    dist_dir = Path(dist_dir)
    json_files = filter(lambda path: DATA_FILENAME_RE.match(path.name), dist_dir.glob('*.json'))
    return [UtterScoreDist.from_json_file(path) for path in json_files]


class PairwiseCorr:

    def __init__(self, distributions):
        self.distributions = distributions
        self.df = pd.DataFrame(
            data=dict((d.name, d.utterance) for d in distributions)
        )
        # lazy computed corr.
        self._corr_data = {}

    @property
    def corr(self, method=None):
        if method is None:
            method = PEARSON
        if method in self._corr_data:
            return self._corr_data[method]
        data = self.df.corr(method)
        return self._corr_data.setdefault(method, data)

    def plot_scatter_matrix(self):
        data = scale_and_sample(self.df)
        return scatter_matrix(data, diagonal='kde')

    def sorted_corr(self, method=None):
        corr = self.corr(method)
        return corr.sort_values(ascending=False)


key_fns = {
    'model_dataset': lambda dist: (dist.model, dist.dataset),
    'metric_dataset': lambda dist: (dist.metric, dist.dataset),
}


class DistGroup:

    def __init__(self, distributions, key_fn):
        if isinstance(key_fn, str):
            key_fn = key_fns[key_fn]
        group = collections.defaultdict(list)
        for dist in distributions:
            key = key_fn(dist)
            group[key].append(dist)
        self.group = {key: PairwiseCorr(dists) for key, dists in group.items()}

    def __getitem__(self, key):
        return self.group[key]

    def __len__(self):
        return len(self.group)
