"""
Plot the barplots for system-level scores.

Each plot is for the scores of one metric and _all_ (model, dataset)s.
The barplots are used in the experiment section of GraduateDesign as a mean to
visualize system-level scores.
"""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

from eval.consts import PLOT_FILENAME, FIGURE_ROOT
from eval.data import seaborn_setup, load_system_score
from eval.normalize import normalize_names_in_df
from eval.utils import make_parent_dirs

NAME = 'barplot'

__version__ = '0.0.1'


def get_output(prefix: Path, metric):
    """
    Make a writable path for output of one metric.

    :param prefix: save plots under this dir.
    :param metric: a plot is uniquely identified by a metric.
    :return: a path with all parent dirs made.
    """
    return make_parent_dirs(prefix / NAME / metric / PLOT_FILENAME)


def plot(df: DataFrame, prefix: Path):
    """
    Plot all the barplots from a df with all system scores.

    :param df: columns are (model, metric, dataset, system).
                Each row is the system score for one model instance on one metric.
    :param prefix: save all plots under this dir.
    :return:
    """
    for metric, df2 in df.groupby('metric'):
        df2 = normalize_names_in_df(df2)
        output = get_output(prefix, metric)
        # catplot() is high-level interface to barplot().
        g = sns.catplot(x='dataset', y='system', hue='model', kind='bar', data=df2)
        # No label, the sticks work fine.
        g.set_xlabels('')
        g.set_ylabels('')
        # No legend title.
        g.fig.legends[0].set_title('')
        logging.info('plotting to {}'.format(output))
        g.savefig(output)
        plt.close('all')


def preprocess() -> DataFrame:
    """
    Load and preprocess the data from the disk.

    :return: a df suitable for plot().
    """
    df = load_system_score(remove_random_model=True)
    # We want the models and datasets in defined order, such as
    # (HRED, LSTM, VHRED).
    df = df.sort_values(['model', 'dataset'])
    return df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    seaborn_setup()
    df = preprocess()
    plot(df=df, prefix=FIGURE_ROOT)
