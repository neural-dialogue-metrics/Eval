import logging
import math
from pathlib import Path

import seaborn as sns
from eval.group import MetricGroup
from pandas import DataFrame
from seaborn import FacetGrid

from eval.consts import PLOT_FILENAME
from eval.normalize import normalize_names_in_df
from eval.data import load_system_score, seaborn_setup
from eval.group import get_col_wrap

logger = logging.getLogger(__name__)

__version__ = '0.0.1'


class SystemScorePlotter:
    def __init__(self, mode, dst_prefix: Path):
        self.mode = mode
        self.dst_prefix = dst_prefix

    def plot(self, df: DataFrame):
        col_wrap = get_col_wrap(MetricGroup.num_groups())
        logger.info('col_wrap {}'.format(col_wrap))
        g = FacetGrid(data=df, col='group', col_wrap=col_wrap)
        # sns.pointplot()
        g.map_dataframe(sns.pointplot, self.mode, 'system', hue='metric')
        g.add_legend()
        output = self.get_output()
        logger.info('plotting to {}'.format(output))
        g.savefig(output)

    def get_output(self):
        parent = self.dst_prefix / 'plot' / 'system' / self.mode
        if not parent.exists():
            parent.mkdir(parents=True)
        return parent / PLOT_FILENAME


if __name__ == '__main__':
    seaborn_setup()
    dst_prefix = Path('/home/cgsdfc/Metrics/Eval/data/v2')
    logging.basicConfig(level=logging.INFO)
    df = load_system_score(prefix=Path('/home/cgsdfc/Metrics/Eval/data/v2/score/db'), remove_random_model=True)
    df = df[df.metric != 'serban_ppl']
    df = MetricGroup().add_group_column_to_df(df)
    df = normalize_names_in_df(df)

    plotter = SystemScorePlotter('model', dst_prefix)
    plotter.plot(df)
