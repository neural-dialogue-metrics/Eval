import logging
from pathlib import Path
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from pandas import Series
from sklearn.preprocessing import scale

from corr.data import DataIndex

NAME = 'distplot'

logger = logging.getLogger(__name__)


def get_output(prefix: Path, triple):
    return prefix / NAME / triple.dataset / triple.model / triple.metric / 'plot.png'


def do_distplot(data, triple, output):
    data = Series(data).transform(scale)
    fig = Figure()
    FigureCanvas(fig)
    ax = fig.add_subplot(111)
    sns.distplot(data, ax=ax)
    ax.set_title('{} of {} on {}'.format(triple.metric, triple.model, triple.dataset))
    fig.savefig(str(output))


def plot(data_index: DataIndex, prefix: Path, force=False):
    for triple in data_index.iter_triples():
        output = get_output(prefix, triple)

        if not output.parent.is_dir():
            output.parent.mkdir(parents=True)

        source: Path = triple.filename
        target = output
        if not force and target.exists() and target.stat().st_mtime > source.stat().st_mtime:
            logger.info('up to date: {}'.format(output))
            continue

        data = data_index.get_data(triple.filename).utterance
        logger.info('plotting to {}'.format(output))
        do_distplot(data, triple, output)
