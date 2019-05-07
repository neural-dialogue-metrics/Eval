import collections
import json
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas import ExcelWriter
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import scale as sklearn_scale

from scipy import stats

from eval.consts import PEARSON
from eval.consts import SAMPLE_SIZE, RANDOM_STATE, SEPARATOR, ALL_METHODS
from eval.repo import get_model, get_dataset

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

    @property
    def parts(self):
        return self.model, self.dataset, self.metric

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
        return sns.distplot(data, fit=stats.norm)


DATA_FILENAME_RE = re.compile(r'\w+-\w+-\w+\.json')


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


def find_all_data_files(dist_dir):
    dist_dir = Path(dist_dir)
    data_files = filter(lambda path: DATA_FILENAME_RE.match(path.name), dist_dir.glob('*.json'))
    return list(data_files)


def load_dists_from_dir(dist_dir):
    return [UtterScoreDist.from_json_file(path) for path in find_all_data_files(dist_dir)]


def load_all(prefix, include='*'):
    logger.info('loading data from %s', prefix)
    data_files = find_all_data_files(prefix)
    data_dicts = [json.load(file.open()) for file in data_files if file.match(include)]
    return pd.DataFrame.from_records(data=data_dicts)


def to_utterance_scores(df: pd.DataFrame):
    data_dict = {s.metric: s.utterance for _, s in df.iterrows()}
    return pd.DataFrame(data=data_dict)


def plot_scatter_matrix(df: pd.DataFrame, title):
    df.sort_index(axis=1, inplace=True)
    scatter_matrix(scale_and_sample(df), diagonal='kde', figsize=(20, 20))
    plt.suptitle(title)


def inter_metric_corr(prefix, output_file):
    # inter-metric correlation on each group of (model, dataset).
    df = load_all(prefix)
    group_by = df.groupby(['model', 'dataset'])
    with ExcelWriter(output_file) as writer:
        for (model, dataset), score in group_by:
            score = to_utterance_scores(score)
            for method in ALL_METHODS:
                logger.info('computing %s on %s-%s', method, model, dataset)
                corr = score.corr(method)
                sheet_name = SEPARATOR.join((model, dataset, method))
                corr.to_excel(writer, sheet_name=sheet_name)
    logger.info('write to %s', output_file)


def inter_metric_scatter_plot(prefix, save_dir=None):
    df = load_all(prefix)
    group_by = df.groupby(['model', 'dataset'])
    for (model, dataset), score in group_by:
        score = to_utterance_scores(score)
        title = f'Metrics measuring {model} trained on {dataset}'
        plot_scatter_matrix(score, title)
        if save_dir is None:
            plt.show()
        else:
            path = Path(save_dir).joinpath(SEPARATOR.join((model, dataset))).with_suffix('.png')
            logger.info('saving figure to %s', path)
            plt.savefig(path)
