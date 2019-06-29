"""
Correlation heatmap of various metrics on per dataset-model.
"""

from eval.consts import PLOT_FILENAME
from seaborn import heatmap
from iconip.utterance import load_model_dataset2_feature, SAVE_ROOT
from eval.data import seaborn_setup
from eval.utils import make_parent_dirs
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import logging
import seaborn as sns
from eval.normalize import normalize_name


def plot_heatmap():
    sns.set(font_scale=0.9, color_codes=True, font='Times New Roman')
    feature = load_model_dataset2_feature()
    for key, value in feature.items():
        output = SAVE_ROOT / 'plot' / 'heatmap' / key[0] / key[1] / PLOT_FILENAME
        logging.info('plotting to {}'.format(output))
        output = make_parent_dirs(output)
        corr = value.corr()  # use pearson
        plt.gcf().subplots_adjust(bottom=0.18, right=1.0)
        heatmap(corr)

        model = normalize_name('model', key[0])
        dataset = normalize_name('dataset', key[1])
        plt.title('Heatmap of Pearson\'s R for {} on {}'.format(model, dataset))
        plt.savefig(output)
        plt.close('all')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    plot_heatmap()
