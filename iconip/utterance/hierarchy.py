"""
Hierarchical clustering with scipy
"""

import logging

import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

from eval.utils import make_parent_dirs
from iconip.utterance import load_all_corr
from pathlib import Path

LINKAGE_METHOD = 'average'

SAVE_ROOT = Path('/home/cgsdfc/ICONIP2019/figure')


def hierarchy_with_corr(corr_matrix: pd.DataFrame):
    # Note DataFrame can be converted to array.
    condensed = squareform(corr_matrix, checks=False, force='tovector')
    condensed = 1 - condensed
    Z = linkage(condensed, method=LINKAGE_METHOD, optimal_ordering=True)
    # Make room for the leaves label.
    plt.gcf().subplots_adjust(right=0.8)
    dendrogram(Z, orientation='left', leaf_label_func=lambda x: corr_matrix.columns[x])


def plot_dendrogram():
    for key, value in load_all_corr().items():
        output = make_parent_dirs(
            SAVE_ROOT / 'plot' / 'hierarchy' / 'v2' / key[0] / key[1] / key[2] / 'plot.eps'
        )
        logging.info('plotting to {}'.format(output))
        try:
            hierarchy_with_corr(value)
        except ValueError as e:
            logging.error(e)
            continue
        plt.savefig(output)
        plt.close('all')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    plt.rcParams['font.family'] = 'Times New Roman'
    plot_dendrogram()
