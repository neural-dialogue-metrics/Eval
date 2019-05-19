import logging
import math
import re
from pathlib import Path

import seaborn as sns
from pandas import DataFrame, Series
from seaborn import FacetGrid

from eval.consts import PLOT_FILENAME
from eval.normalize import normalize_names_in_df
from eval.data import load_system_score, seaborn_setup

logger = logging.getLogger(__name__)

__version__ = '0.0.1'


def exact(string):
    return lambda s: s == string


def contains(string):
    return lambda s: string in s


def re_match(pattern):
    return lambda s: re.match(pattern, s)


group_matchers = {
    'Embedding': contains('embedding_based'),
    'BLEU': contains('bleu'),
    'Distinct-N': re_match(r'distinct_\d'),
    'ROUGE-N': re_match(r'rouge_\d'),
    'ROUGE-L/W': re_match(r'rouge_[lw]'),
    'ADEM': exact('adem'),
    '#words': exact('utterance_len'),
    'METEOR': exact('meteor'),
}


def assign_group(df: DataFrame):
    group_values = []
    for mc in df.metric.values:
        for name, matcher in group_matchers.items():
            if matcher(mc):
                group_values.append(name)
                break
        else:
            raise ValueError('unable to match {} against one of {}'.format(
                mc, tuple(group_matchers.keys())))
    return Series(group_values)


def check_group(df: DataFrame):
    errs = 0
    for metric, group in df.loc[:, ['metric', 'group']].values:
        logger.info('{} => {}'.format(metric, group))
        matcher = group_matchers[group]
        if not matcher(metric):
            errs += 1
            logger.error('mismatched metric {} to group {}'.format(metric, group))
    if errs:
        raise ValueError('there are mismatches')


class SystemScorePlotter:
    def __init__(self, mode, dst_prefix: Path):
        self.mode = mode
        self.dst_prefix = dst_prefix

    def plot(self, df: DataFrame):
        col_wrap = int(math.sqrt(len(group_matchers) + 1))
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
    df = load_system_score(
        prefix=Path('/home/cgsdfc/Metrics/Eval/data/v2/score/db'),
        remove_random_model=True,
    )
    df = df[df.metric != 'serban_ppl'].reset_index()
    df = df.assign(group=assign_group)
    df = normalize_names_in_df(df)

    plotter = SystemScorePlotter('model', dst_prefix)
    plotter.plot(df)
