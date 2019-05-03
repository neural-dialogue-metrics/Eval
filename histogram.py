from eval.correlate import UtterScoreDist
import argparse
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    args = parser.parse_args()
    dist = UtterScoreDist.from_json_file(args.input)
    dist.plot_dist()
    plt.show()
