import argparse
import logging
from graduate.utils import inter_metric_corr, inter_metric_scatter_plot

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix')
    parser.add_argument('-x', '--excel', help='output excel filename')
    parser.add_argument('--save-dir')
    parser.add_argument('--corr', action='store_true', help='perform inter-metric correlation analysis')
    parser.add_argument('--scatter-plot', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.corr:
        inter_metric_corr(args.prefix, args.excel)
    elif args.scatter_plot:
        inter_metric_scatter_plot(args.prefix, args.save_dir)
