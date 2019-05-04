import argparse
import logging

from pathlib import Path
from eval.pairwise import compute_corr_and_plot_scatter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-u')
    parser.add_argument('-v')
    parser.add_argument('-p')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    compute_corr_and_plot_scatter(
        u=Path(args.u),
        v=Path(args.v),
        output_dir=Path(args.p),
    )
