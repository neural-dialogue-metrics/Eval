from pathlib import Path

from eval.data import UtterScoreDist
import argparse
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('-q', '--quiet', action='store_true', help='save the figure quietly')
    parser.add_argument('-p', '--prefix', help='output to this prefix')
    parser.add_argument('-f', '--format', default='.png')
    args = parser.parse_args()
    input_file = Path(args.input)

    dist = UtterScoreDist.from_json_file(input_file)
    dist.plot_dist()

    if args.quiet:
        prefix = args.prefix or '.'
        output_file = Path(prefix).joinpath(input_file.name).with_suffix(args.format)
        plt.savefig(output_file)
    else:
        plt.show()
