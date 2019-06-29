from iconip.utterance import load_model_dataset2_feature, SAVE_ROOT, load_corr_matrix
from eval.data import seaborn_setup
from eval.utils import make_parent_dirs
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import logging
import seaborn as sns
from eval.normalize import normalize_name
from seaborn import clustermap


def plot_clustermap():
    pass


if __name__ == '__main__':
    test_key = ('vhred', 'opensub', 'pearson')
    test_corr = load_corr_matrix(*test_key)
    print(test_corr)
