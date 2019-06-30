import logging
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame, Series
from seaborn import FacetGrid

from eval.consts import PLOT_FILENAME
from eval.normalize import normalize_names_in_df
from eval.data import UtterScoreDist, load_score_db_index

NAME = 'distplot_grid'

logger = logging.getLogger(__name__)

__version__ = '0.0.1'


def get_output(prefix: Path, metric):
    output = prefix / NAME / metric / PLOT_FILENAME
    if not output.parent.exists():
        output.parent.mkdir(parents=True)
    return output


def do_distplot(ax, data: UtterScoreDist):
    sns.distplot(data.utterance, ax=ax)
    ax.set_title('{} on {}'.format(data.model, data.dataset))


def distplot_wrapper(filename: Series, **kwargs):
    filename = filename.values[0]
    data = UtterScoreDist(filename, normalize_names=True, scale_values=True)
    sns.distplot(data.utterance, **kwargs)
    plt.title('{} on {}'.format(data.model, data.dataset))


def plot(data_index: DataFrame, prefix: Path):
    data_index = data_index.sort_values(by=['metric', 'model', 'dataset'])

    for metric, df2 in data_index.groupby('metric'):
        df2 = df2.reset_index()
        output = get_output(prefix, metric)
        df2 = normalize_names_in_df(df2)
        g = FacetGrid(df2, row='model', col='dataset')

        g.map(distplot_wrapper, 'filename')
        g.set_xlabels('')
        g.set_titles('{row_name} on {col_name}')
        logger.info('plotting to {}'.format(output))
        g.savefig(output)
        plt.close('all')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    sns.set(color_codes=True, font='Times New Roman')

    dst_prefix = Path('/home/cgsdfc/Metrics/Eval/data/v2/plot')

    df = load_score_db_index(
        Path('/home/cgsdfc/Metrics/Eval/data/v2/score/db')
    )

    plot(df, dst_prefix)
