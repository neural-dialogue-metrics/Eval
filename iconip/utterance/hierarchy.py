"""
Hierarchical clustering with scipy
"""

from scipy.cluster.hierarchy import linkage, dendrogram
from iconip.utterance import load_feature, SAVE_ROOT, normalize_key
import logging
import matplotlib.pyplot as plt
from eval.utils import make_parent_dirs


def plot_dendrogram():
    for key, value in load_feature().items():
        output = make_parent_dirs(
            SAVE_ROOT / 'plot' / 'hierarchy' / key[0] / key[1] / 'plot.pdf'
        )
        logging.info('plotting to {}'.format(output))
        try:
            Z = linkage(value.transpose(), metric='correlation')
        except ValueError as e:
            logging.error(e)
            continue
        plt.gcf().subplots_adjust(right=0.8)
        dendrogram(Z,
                   orientation='left', leaf_label_func=lambda x: value.columns[x])
        plt.title('Hierarchical cluster of metrics of ({}, {})'.format(
            *normalize_key(key)
        ))
        plt.savefig(output)
        plt.close('all')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    plt.rcParams['font.family'] = 'Times New Roman'
    plot_dendrogram()
