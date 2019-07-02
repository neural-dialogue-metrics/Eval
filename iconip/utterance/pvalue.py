"""
Compute the p-value for all correlation methods.

The smaller the p-value, the higher the significance.
"""
import logging
import json
import pickle
from itertools import combinations, product
from collections import defaultdict
import numpy as np
from pandas import DataFrame
from scipy.stats import pearsonr, spearmanr, kendalltau

from eval.utils import make_parent_dirs
from iconip.utterance import load_all_scores, SAVE_ROOT


def compute_pvalue(scores: DataFrame, method):
    """
    Compute the p-value of the given correlation method.
    The correlation is *between two metrics*, not between *two examples*.

    :param scores: a df of score matrix.
    :return: a df of symmetric matrix of p-value.
    """

    items = defaultdict(dict)
    for x, y in product(scores.columns, repeat=2):
        corr, pvalue = method(scores[x], scores[y])
        if np.isnan(pvalue):
            pvalue = 0  # NaN handling?
        items[x][y] = pvalue

    return items


def compute():
    method_dict = {
        'pearson': pearsonr,
        'spearman': spearmanr,
        'kendall': kendalltau,
    }
    for key, value in load_all_scores().items():
        for method_name, method in method_dict.items():
            output = make_parent_dirs(SAVE_ROOT / 'pvalue' / method_name / key[0] / key[1] / 'data.json')
            logging.info('computing pvalue for {}, method={}'.format(key, method_name))
            try:
                pvalue = compute_pvalue(value, method)
                json.dump(pvalue, output.open('w'))
            except OverflowError as e:
                logging.warning(e)


THESHOLD = 0.05


def find_high_value(method='pearson'):
    files = (SAVE_ROOT / 'pvalue').rglob('{}/**/data.json'.format(method))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    compute()
