import itertools
import logging
import traceback
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.preprocessing import scale

from corr.data import DataIndex

warnings.filterwarnings('ignore', r'JointGrid annotation is deprecated')
from corr.utils import UtterScoreDist

from corr.consts import *

NAME = 'jointplot'
logger = logging.getLogger(__name__)


def get_output(prefix: Path, x, y, mode):
    if x.dataset != y.dataset:
        raise ValueError('incompatible dataset: {} and {}'.format(x.dataset, y.dataset))
    ds = x.dataset
    prefix = prefix / NAME / ds
    if mode == MODE_METRIC:
        if x.metric != y.metric:
            raise ValueError('metric must be the same on this mode')
        return prefix / 'metric' / x.metric / '{}-{}'.format(x.model, y.model) / PLOT_FILENAME
    if mode == MODE_MODEL:
        if x.model != y.model:
            raise ValueError('model must be the same on this mode')
        return prefix / 'model' / x.model / '{}-{}'.format(x.metric, y.metric) / PLOT_FILENAME
    raise ValueError('invalid mode {}'.format(mode))


def cn_2(iterable):
    return itertools.combinations(iterable, 2)


def do_plot(x: UtterScoreDist, y: UtterScoreDist, mode: str, output):
    joint_grid = sns.jointplot(
        x=scale(x.utterance),
        y=scale(y.utterance),
    )
    joint_grid.annotate(func=pearsonr, stat="Pearson's R")
    if mode == MODE_MODEL:
        joint_grid.fig.suptitle('{model} on {dataset}'.format(model=x.model, dataset=x.dataset))
        joint_grid.set_axis_labels(xlabel=x.metric, ylabel=y.metric)
    else:
        joint_grid.fig.suptitle('{metric} on {dataset}'.format(metric=x.metric, dataset=x.dataset))
        joint_grid.set_axis_labels(xlabel=x.model, ylabel=y.model)
    joint_grid.savefig(str(output))
    logger.info('plotting to {}'.format(output))
    plt.close()


def plot(data_index: DataIndex, prefix: Path, force=False):
    def plot_for_mode(mode):
        logger.info('plotting for mode {}'.format(mode))
        for key, data in data_index.index.groupby(['dataset', mode]):
            triples = list(data.itertuples(index=False, name='Triple'))
            pairs = cn_2(triples)
            for x, y in pairs:
                output = get_output(prefix, x, y, mode)
                if not output.parent.is_dir():
                    output.parent.mkdir(parents=True)

                sources = [x.filename, y.filename]
                if not force and output.exists() and all(
                        output.stat().st_mtime > src.stat().st_mtime for src in sources):
                    logger.info('up to date: {}'.format(output))
                    continue
                try:
                    do_plot(x=data_index.get_data(x.filename),
                            y=data_index.get_data(y.filename), mode=mode, output=output)
                except Exception:
                    traceback.print_exc()
                    logging.info('x: {}'.format(x))
                    logging.info('y: {}'.format(y))

    plot_for_mode(MODE_MODEL)
    plot_for_mode(MODE_METRIC)
