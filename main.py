import argparse

from eval.config_parser import load_config
from eval.engine import Engine

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='path to config file (default to builtin config)')
    parser.add_argument('-p', '--prefix', help='path to output directory')
    parser.add_argument('-f', '--force', action='store_true', help='force to overwrite existing files (default is to skip)')
    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
    else:
        from repo import config

    engine = Engine(config, args.prefix, args.force)
    engine.run()
