from iconip.utterance import load_model_dataset2_feature, SAVE_ROOT, load_corr_matrix
from eval.data import seaborn_setup
from eval.consts import PLOT_FILENAME
from eval.utils import make_parent_dirs
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import logging
import seaborn as sns
from eval.normalize import normalize_name
from seaborn import clustermap
from pandas import DataFrame
import numpy as np

THRESHOLD = 1000


def patch_null_columns(df: DataFrame):
    for col in df.columns:
        column = df[col]
        if not column.any():
            logging.warning('column {} is all zero'.format(col))
            df[col] = 1
    return df


def plot_clustermap():
    seaborn_setup()
    for key, value in load_model_dataset2_feature().items():
        output = make_parent_dirs(SAVE_ROOT / 'plot' / 'clustermap' / key[0] / key[1] / PLOT_FILENAME)
        if len(value) > THRESHOLD:
            logging.info('value too large: {}'.format(len(value)))
            value = value.sample(n=THRESHOLD)
        # value = patch_null_columns(value)
        logging.info('plotting to {}'.format(output))
        # scale the column. cluster the column.
        try:
            g = clustermap(value, metric='correlation', vmin=-1, vmax=1, z_score=1, row_cluster=False)
        except Exception as e:
            logging.warning('error: {}'.format(e))
        else:
            g.savefig(output)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    plot_clustermap()
