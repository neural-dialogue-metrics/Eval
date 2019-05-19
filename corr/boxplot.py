from pathlib import Path

import seaborn as sns
from eval.consts import PLOT_FILENAME
from eval.data import get_schema_name, load_system_score
from eval.utils import make_parent_dirs
from pandas import DataFrame

NAME = get_schema_name(__file__)

__version__ = '0.0.1'


def get_output(prefix: Path, metric):
    return make_parent_dirs(prefix / NAME / metric / PLOT_FILENAME)


def preprocess() -> DataFrame:
    df = load_system_score(remove_random_model=True, normalize_name=True)
    df = df.sort_values(['metric', 'model', 'dataset'])
    return df


def plot(df: DataFrame):
    g = sns.catplot(x='dataset', y='system', hue='model', kind='box', data=df)
    g.set_xlabels('')
    g.set_ylabels('')
    g.fig.legends[0].set_title('')
    return g
