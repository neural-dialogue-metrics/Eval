import argparse

from eval.config import config
from eval.engine import Engine

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', help='path to output files')
    args = parser.parse_args()

    engine = Engine(config, args.prefix)
    engine.run()
