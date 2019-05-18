import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame, Series

from corr.consts import PLOT_FILENAME
from corr.normalize import normalize_name, normalize_names_in_df
from corr.utils import UtterScoreDist, load_filename_data
from seaborn import FacetGrid

NAME = 'distplot_grid'

logger = logging.getLogger(__name__)


def get_output(prefix: Path, metric):
    output = prefix / NAME / metric / PLOT_FILENAME
    if not output.parent.exists():
        output.parent.mkdir(parents=True)
    return output


def do_distplot(ax, data: UtterScoreDist):
    sns.distplot(data.utterance, ax=ax)
    ax.set_title('{} on {}'.format(data.model, data.dataset))


def plot(data_index: DataFrame, prefix: Path):
    data_index = data_index.sort_values(by=['metric', 'model', 'dataset'])
    plt.tight_layout()

    for metric, df2 in data_index.groupby('metric'):
        df2 = df2.reset_index()
        output = get_output(prefix, metric)
        df2 = normalize_names_in_df(df2)
        g = FacetGrid(df2, row='model', col='dataset')

        def distplot_wrapper(filename: Series, **kwargs):
            filename = filename.values[0]
            data = UtterScoreDist(filename, scale=True, normalize=True)
            sns.distplot(data.utterance, **kwargs)
            plt.title('{} on {}'.format(data.model, data.dataset))

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

    df = load_filename_data(
        Path('/home/cgsdfc/Metrics/Eval/data/v2/score/db')
    )

    plot(df, dst_prefix)
