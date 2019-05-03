import argparse

from eval.config_parser import load_config
from eval.engine import Engine

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='path to config file (default to builtin config)')
    parser.add_argument('-p', '--prefix', help='path to output directory')
    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
    else:
        from eval.config import config

    engine = Engine(config, args.prefix)
    engine.run()
