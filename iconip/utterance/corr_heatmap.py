"""
Correlation heatmap of various metrics on per dataset-model.
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


def plot_heatmap():
    sns.set(font_scale=0.9, style='white', font='Times New Roman')
    for (method, model, dataset), corr in load_all_corr().items():
        output = PLOT_ROOT / 'plot' / 'heatmap' / 'v3' / method / model / dataset / PLOT_FILENAME
        logging.info('plotting to {}'.format(output))
        output = make_parent_dirs(output)

        # Plotting logic.
        plt.gcf().subplots_adjust(bottom=0.18, right=1.0)
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        heatmap(
            corr, center=0, cmap=cmap, vmax=1, vmin=-1,
            square=True, linewidth=0.5, mask=mask,
            cbar_kws={
                'shrink': 0.5
            })
        plt.savefig(output)
        plt.close('all')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    plot_heatmap()
