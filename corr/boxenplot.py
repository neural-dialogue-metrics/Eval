import logging
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from eval.consts import PLOT_FILENAME, FIGURE_ROOT
from eval.data import load_system_score, seaborn_setup
from eval.normalize import normalize_names_in_df
from eval.utils import make_parent_dirs
from pandas import DataFrame

# boxen or box
NAME = 'box'

__version__ = '0.2.0'


def get_output(prefix: Path, metric):
    return make_parent_dirs(prefix / NAME / metric / PLOT_FILENAME)


def preprocess() -> DataFrame:
    df = load_system_score(remove_random_model=True)
    df = df.sort_values(['metric', 'model', 'dataset'])
    return df


def do_plot(df: DataFrame, x_var):
    g = sns.catplot(x=x_var, y='system', kind=NAME, data=df, ci='sd')
    g.set_xlabels('')
    g.set_ylabels('')
    return g


x_vars = ('dataset', 'model')


def plot(df: DataFrame, prefix: Path):
    for metric, df2 in df.groupby('metric'):
        df2 = normalize_names_in_df(df2)
        for x_var in x_vars:
            g = do_plot(df2, x_var)
            output = make_parent_dirs(prefix / NAME / x_var / metric / PLOT_FILENAME)
            logging.info('plotting to {}'.format(output))
            g.savefig(output)
            plt.close('all')


if __name__ == '__main__':
    seaborn_setup()
    logging.basicConfig(level=logging.INFO)
    df = preprocess()
    plot(
        df=df,
        prefix=Path(FIGURE_ROOT)
    )
