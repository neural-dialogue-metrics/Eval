"""
Correlation heatmap of various metrics for each instance.

This a direct visualization of the correlation matrix. Each number of the matrix is replaced by a colored
cell, reflecting the magnitude of the value. Specifically, possitive correlation uses warm color while
negative correlation uses cold color. Zero correlation uses white.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from seaborn import heatmap

from eval.consts import PLOT_FILENAME
from eval.utils import make_parent_dirs
from iconip.utterance import PLOT_ROOT, load_all_corr

# Generate a red-blue-white palette.
# See http://seaborn.pydata.org/generated/seaborn.diverging_palette.html
cmap = sns.diverging_palette(240, 10, as_cmap=True)
VERSION = 'v4'


def plot_heatmap():
    """
    Plots all the heatmaps.

    :return:
    """
    sns.set(font_scale=0.9, style='white', font='Times New Roman')
    for (method, model, dataset), corr in load_all_corr().items():
        output = PLOT_ROOT / 'plot' / 'heatmap' / VERSION / method / model / dataset / PLOT_FILENAME
        logging.info('plotting to {}'.format(output))
        output = make_parent_dirs(output)

        # Plotting logic.
        plt.gcf().subplots_adjust(bottom=0.18, right=1.0)  # Fit all labels in.
        # Mask is not used.
        # mask = np.zeros_like(corr, dtype=np.bool)
        # mask[np.triu_indices_from(mask)] = True
        ax = heatmap(
            corr, center=0, cmap=cmap, vmax=1, vmin=-1,
            square=True, linewidth=0.5,
            cbar_kws={
                'shrink': 0.5
            })
        ax.set_aspect('equal')
        plt.savefig(output, bbox_inches='tight')
        plt.close('all')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    plot_heatmap()
