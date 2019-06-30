"""
Hierarchical clustering with scipy.

For each instance, a matrix of example-level scores are defined, from which inter-metric correlation
can be computed. To show the relationship among the metrics, a clustering algorithm is applied to them,
based on the correlation matrix.

The clustering is done with precomputed correlation matrices of three different methods, namely
pearson, spearman and kendall. As the correlation measures similarity, it is converted to some form of
distance by $1 - corr$ as in scipy. The dense matrix is transformed to condensed vector by `squareform()`.
The plots are optimized for readability and economic of page space.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

from eval.utils import make_parent_dirs
from iconip.utterance import load_all_corr, PLOT_ROOT

LINKAGE_METHOD = 'average'
# Version for the plots.
VERSION = 'v3'
PLOT_FILENAME = 'plot.pdf'


def hierarchy_with_corr(corr_matrix: pd.DataFrame):
    """
    Perform hierarchical clustering and plot dendrograms with `corr_matrix`.

    :param corr_matrix: A df of precomputed correlation data in dense format.
    :return:
    """
    # Note DataFrame can be converted to array.
    condensed = squareform(corr_matrix, checks=False, force='tovector')
    # For it to be a *distance*.
    condensed = 1 - condensed
    Z = linkage(condensed, method=LINKAGE_METHOD, optimal_ordering=True)
    # Make room for the leaves label.
    plt.gcf().subplots_adjust(right=0.8)
    dendrogram(Z, orientation='left', leaf_label_func=lambda x: corr_matrix.columns[x])


def plot_dendrogram():
    """
    Plots all the dendrograms.

    :return:
    """
    for key, value in load_all_corr().items():
        output = make_parent_dirs(
            PLOT_ROOT / 'plot' / 'hierarchy' / VERSION / key[0] / key[1] / key[2] / PLOT_FILENAME
        )
        logging.info('plotting to {}'.format(output))
        try:
            hierarchy_with_corr(value)
        except Exception as e:
            logging.error(e)
            continue
        # Note `bbox_inches='tight'` is the key to making margin-minimized figures.
        plt.savefig(output, bbox_inches='tight')
        plt.close('all')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    plt.rcParams['font.family'] = 'Times New Roman'
    plot_dendrogram()
