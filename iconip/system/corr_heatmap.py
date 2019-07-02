"""
Plot the correlation heatmaps for various instances and methods.
"""

from iconip import CorrHeatmapPlotter
from iconip.system import SAVE_ROOT
from eval.utils import make_parent_dirs
from pandas import DataFrame
import logging
import pandas


def plot_heatmaps():
    p = CorrHeatmapPlotter()
    root = (SAVE_ROOT / 'corr')
    for path in root.rglob('*.json'):
        basename = path.relative_to(root).with_name('plot.pdf')
        output = make_parent_dirs(SAVE_ROOT / 'plot' / 'heatmap' / basename)
        corr = pandas.read_json(path)
        p.plot(corr, output, annot=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    plot_heatmaps()
