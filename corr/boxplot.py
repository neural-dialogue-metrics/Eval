"""
Mostly similar to barplot.py, with barplot changed to boxplot.

The boxplot reduces one dimension of the data to its central tendency.
It sacrisfies the accuracy of one dimension to emphasize the overall trends of the other dimension.
For example, if the model-dim is to be emphasized, then the dataset-dim will become a scalar.
A plot like this will list the stretch of a score of different datasets side-by-side for the ease of comparison.
In the paper of my graduate design, they are used as the complementary materials to more accurate barplot to highlight
distributions of system scores along the model and the dataset dimensions, respectively.
Note a boxplot is also uniquely identified by a metric like a barplot.
"""
import logging
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from pandas import DataFrame

from eval.consts import PLOT_FILENAME, FIGURE_ROOT
from eval.data import load_system_score, seaborn_setup
from eval.normalize import normalize_names_in_df
from eval.utils import make_parent_dirs

NAME = 'boxplot'

__version__ = '0.2.0'


def preprocess() -> DataFrame:
    """
    Load system scores.

    :return: a regularized df of all scores.
    """
    df = load_system_score(remove_random_model=True)
    df = df.sort_values(['metric', 'model', 'dataset'])
    return df


def do_plot(df: DataFrame, x_var):
    """
    Perform a barplot on a column specified by `x_var`.

    :param df: holds system scores of all instances.
    :param x_var: the name of the column to be plotted.
    :return: FacetGrid.
    """
    # Use std as the length of error bars.
    g = sns.catplot(x=x_var, y='system', kind='box', data=df, ci='sd')
    # There is no legend.
    g.set_xlabels('')
    g.set_ylabels('')
    return g


# The possible dimensions.
X_VARS = ('dataset', 'model')


def plot(df: DataFrame, prefix: Path):
    """
    Plot boxplots for all metrics.

    :param df: for all scores.
    :param prefix: results are saved here.
    :return:
    """
    for metric, df2 in df.groupby('metric'):
        df2 = normalize_names_in_df(df2)
        for x_var in X_VARS:
            g = do_plot(df2, x_var)
            output = make_parent_dirs(prefix / NAME / x_var / metric / PLOT_FILENAME)
            logging.info('plotting to {}'.format(output))
            g.savefig(output)
            plt.close('all')


if __name__ == '__main__':
    seaborn_setup()
    logging.basicConfig(level=logging.INFO)
    df = preprocess()
    plot(df=df, prefix=Path(FIGURE_ROOT))
