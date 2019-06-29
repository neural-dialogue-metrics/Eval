"""
Correlation heatmap of various metrics on per dataset-model.
"""

from eval.consts import PLOT_FILENAME
from seaborn import heatmap
from iconip.utterance import load_feature, SAVE_ROOT, normalize_key, load_all_corr
from eval.data import seaborn_setup
from eval.utils import make_parent_dirs
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import logging
import seaborn as sns


def plot_heatmap():
    sns.set(font_scale=0.9, color_codes=True, font='Times New Roman')
    for (method, model, dataset), corr in load_all_corr().items():
        output = SAVE_ROOT / 'plot' / 'heatmap' / 'v2' / method / model / dataset / PLOT_FILENAME
        logging.info('plotting to {}'.format(output))
        output = make_parent_dirs(output)
        # Plotting logic.
        plt.gcf().subplots_adjust(bottom=0.18, right=1.0)
        heatmap(corr, center=0)
        plt.title('{} Correlation of Metrics on ({}, {})'.format(method.capitalize(),
                                                                 *normalize_key((model, dataset))))
        plt.savefig(output)
        plt.close('all')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    plot_heatmap()
