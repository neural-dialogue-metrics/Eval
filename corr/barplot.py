import seaborn as sns
import logging

from eval.data import seaborn_setup, get_schema_name, load_system_score
from pathlib import Path

from eval.normalize import normalize_names_in_df
from eval.utils import make_parent_dirs
from eval.consts import PLOT_FILENAME
from pandas import DataFrame

NAME = get_schema_name(__file__)

__version__ = '0.0.1'


def get_output(prefix: Path, metric):
    return make_parent_dirs(prefix / NAME / metric / PLOT_FILENAME)


def plot(df: DataFrame, prefix: Path):
    for metric, df2 in df.groupby('metric'):
        df2 = normalize_names_in_df(df2)
        output = get_output(prefix, metric)
        g = sns.catplot(x='dataset', y='system', hue='model', kind='bar', data=df2)
        g.set_xlabels('')
        g.set_ylabels('')
        g.fig.legends[0].set_title('')
        logging.info('plotting to {}'.format(output))
        g.savefig(output)


def preprocess() -> DataFrame:
    df = load_system_score(remove_random_model=True)
    df = df.sort_values(['model', 'dataset'])
    return df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    seaborn_setup()
    df = preprocess()
    plot(
        df=df,
        prefix=Path('/home/cgsdfc/GraduateDesign/figure')
    )
