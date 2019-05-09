"""
A random model -- shuffle the ground truth as output
"""
from pathlib import Path

import numpy as np
import argparse
import logging

from eval.config_parser import load_config, parse_dataset
from eval.consts import RANDOM_RESULT_ROOT, OUTPUT_FILENAME

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config')
    parser.add_argument('-p', '--prefix')
    args = parser.parse_args()
    prefix = args.prefix or Path(RANDOM_RESULT_ROOT)
    logging.basicConfig(level=logging.INFO)

    if args.config is None:
        from eval.repo import all_datasets as datasets
    else:
        datasets = load_config(args.config)['datasets']

    datasets = parse_dataset(datasets)
    for ds in datasets:
        logging.info('dataset: {}'.format(ds.references))
        references = Path(ds.references).open().readlines()
        np.random.shuffle(references)
        output = prefix.joinpath(ds.name).joinpath(OUTPUT_FILENAME)
        if not output.parent.exists():
            output.parent.mkdir(parents=True)
        output.open('w').writelines(references)
