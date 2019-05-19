import logging
from pathlib import Path

import seaborn as sns
from eval.consts import PLOT_FILENAME, DATA_V2_ROOT
from eval.data import UtterScoreDist, load_filename_data, seaborn_setup
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from pandas import DataFrame

NAME = 'distplot'

logger = logging.getLogger(__name__)

__version__ = '0.0.1'


def get_output(prefix: Path, triple):
    return prefix / NAME / triple.dataset / triple.model / triple.metric / PLOT_FILENAME


def do_distplot(data: UtterScoreDist, output):
    fig = Figure()
    FigureCanvas(fig)
    ax = fig.add_subplot(111)
    sns.distplot(data.utterance, ax=ax)
    ax.set_title('{} of {} on {}'.format(data.metric, data.model, data.dataset))
    logger.info('plotting to {}'.format(output))
    fig.savefig(str(output))


def plot(df: DataFrame, prefix: Path):
    for triple in df.itertuples(index=False):
        output = get_output(prefix, triple)
        if not output.parent.is_dir():
            output.parent.mkdir(parents=True)

        data = UtterScoreDist(triple.filename, scale=True, normalize=True)
        do_distplot(data, output)


if __name__ == '__main__':
    seaborn_setup()
    df = load_filename_data()
    prefix = Path(DATA_V2_ROOT).joinpath('plot')

    plot(df, prefix)
