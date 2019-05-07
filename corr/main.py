import matplotlib.pyplot as plt
import logging
from pathlib import Path
import seaborn as sns
from corr.data import DataIndex
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir')
    parser.add_argument('-p', '--prefix')
    parser.add_argument('-f', '--force', action='store_true')
    logging.basicConfig(level=logging.INFO)
    sns.set(color_codes=True)

    args = parser.parse_args()
    from corr.distplot import plot

    data_index = DataIndex(args.data_dir)
    logging.info('backend: {}'.format(plt.get_backend()))
    plot(data_index, Path(args.prefix), args.force)
