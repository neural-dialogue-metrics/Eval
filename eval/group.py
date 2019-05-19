import re

from pandas import DataFrame, Series
from system_score_plot import logger


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
