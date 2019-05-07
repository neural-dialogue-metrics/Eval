import matplotlib

matplotlib.use('Agg')
from pandas import Series

import seaborn as sns
import logging
from matplotlib.axes import Axes
from corr.data import DataIndex
from pathlib import Path
from sklearn.preprocessing import scale

NAME = 'distplot'

logger = logging.getLogger(__name__)


def get_output(prefix: Path, triple):
    return prefix / NAME / triple.dataset / triple.model / triple.metric / 'plot.png'


def do_distplot(data, triple):
    # Figure out a good set of params.
    axes: Axes = sns.distplot(data)
    axes.set_title('{} of {} on {}'.format(triple.metric, triple.model, triple.dataset))
    return axes


def plot(data_index: DataIndex, prefix: Path, force=False):
    for triple in data_index.iter_triples():
        output = get_output(prefix, triple)

        if not output.parent.is_dir():
            output.parent.mkdir(parents=True)

        source: Path = triple.filename
        target = output
        if target.exists() and target.stat().st_mtime > source.stat().st_mtime:
            if force:
                logger.info('force remake {}'.format(output))
            else:
                logger.info('up to date: {}'.format(output))
                continue

        data = data_index.get_data(triple.filename)
        data = Series(data.utterance).transform(scale)

        logger.info('plotting to {}'.format(output))
        axes: Axes = do_distplot(data, triple)
        axes.get_figure().savefig(output.open('wb'))
