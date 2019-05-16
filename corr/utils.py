import json
import logging
import re
from pathlib import Path

import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import scale as sklearn_scale

from corr.consts import SAMPLE_SIZE, RANDOM_STATE
from eval.consts import SEPARATOR

logger = logging.getLogger(__name__)

DATA_FILENAME_RE = re.compile(r'\w+-\w+-\w+\.json')


def scale_and_sample(frame: pd.DataFrame):
    return frame.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).transform(sklearn_scale)


class DataIndex:

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir).absolute()
        self._index = None
        self._cache = {}

    @property
    def index(self):
        if self._index is None:
            self._index = load_filename_data(self.data_dir)
        return self._index

    def iter_triples(self):
        return self.index.itertuples(index=False, name='Triple')

    def get_data(self, path, scale=False):
        if path in self._cache:
            return self._cache[path]
        return self._cache.setdefault(path, UtterScoreDist.from_json_file(path, scale))


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

    def __init__(self, model, dataset, metric, system, utterance, scale=False):
        super(UtterScoreDist, self).__init__(model, dataset, metric)
        self.system = system
        self.scaled = scale
        if scale:
            utterance = sklearn_scale(utterance)
        self.utterance = utterance

    @classmethod
    def from_json_file(cls, filename, scale=False):
        data = json.load(open(filename))
        return cls(**data, scale=scale)


def find_all_data_files(dir):
    dir = Path(dir)
    data_files = filter(lambda path: DATA_FILENAME_RE.match(path.name), dir.glob('*.json'))
    return list(data_files)


def is_fully_substituted(url):
    return re.search(r'<[\w_\d]+>', url) is None


def load_filename_data(data_dir):
    logger.info('loading filename data from {}'.format(data_dir))
    data_files = find_all_data_files(data_dir)

    def parse(filename: Path):
        model, dataset, metric = filename.stem.split(SEPARATOR)
        return locals()

    return DataFrame.from_records([parse(p) for p in data_files])


def substitute_url(url: str, check_full=False, **kwargs):
    for name, value in kwargs.items():
        url = url.replace('<{}>'.format(name), value)
    if check_full and not is_fully_substituted(url):
        raise ValueError('url not fully substituted: {}'.format(url))
    return url


def remake_needed(target: Path, *sources, force=False):
    if force:
        return True
    if not target.exists():
        return True
    for src in sources:
        if not src.exists():
            raise ValueError('cannot remake with {}'.format(src))
        if target.stat().st_mtime < src.stat().st_mtime:
            return True
    logger.info('{} is up to date'.format(target))
    return False


def get_plots(name):
    locals_vars = {}
    try:
        exec('from corr.{} import plot'.format(name), locals_vars, locals_vars)
    except ImportError as e:
        raise ValueError('invalid plot {}'.format(name)) from e
    return locals_vars['plot']


def plot_main():
    import argparse
    import seaborn as sns
    import matplotlib.pyplot as plt
    from corr import all_plotters

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir', help='where to look for score data')
    parser.add_argument('-p', '--prefix', help='where to store the plots')
    parser.add_argument('-f', '--force', action='store_true', help='remake everything regardless of timestamp')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    sns.set(color_codes=True)
    sns.set(font='Times New Roman')

    data_index = DataIndex(args.data_dir)
    for name in all_plotters:
        logging.info('running {}'.format(name))
        plot_fn = get_plots(name)
        plot_fn(data_index, Path(args.prefix), force=args.force)

    logging.info('backend: {}'.format(plt.get_backend()))
    logging.info('all done')
