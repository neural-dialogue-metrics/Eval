import logging
import re
from pathlib import Path

from eval.data import DataIndex

logger = logging.getLogger(__name__)

DATA_FILENAME_RE = re.compile(r'\w+-\w+-\w+\.json')


def get_plots(name):
    locals_vars = {}
    try:
        exec('from corr.{} import plot'.format(name), locals_vars, locals_vars)
    except ImportError as e:
        raise ValueError('invalid plot {}'.format(name)) from e
    return locals_vars['plot']


def plot_main():
    import argparse
    import seaborn as sns
    import matplotlib.pyplot as plt
    from corr import all_plotters

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir', help='where to look for score data')
    parser.add_argument('-p', '--prefix', help='where to store the plots')
    parser.add_argument('-f', '--force', action='store_true', help='remake everything regardless of timestamp')
    parser.add_argument('-x', '--select', help='run a specific plot instead of all')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    sns.set(color_codes=True)
    sns.set(font='Times New Roman')

    data_index = DataIndex(args.data_dir)
    if args.select:
        all_plotters = [args.select]

    for name in all_plotters:
        logging.info('running {}'.format(name))
        plot_fn = get_plots(name)
        plot_fn(data_index, Path(args.prefix), force=args.force)

    logging.info('backend: {}'.format(plt.get_backend()))
    logging.info('all done')
