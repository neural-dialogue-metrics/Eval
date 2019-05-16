"""
A random model -- shuffle the ground truth as output
"""
import argparse
import logging
from pathlib import Path

import numpy as np

from eval.config_parser import load_config, parse_dataset
from eval.consts import RANDOM_MODEL_ROOT
from eval.repo import get_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Run the random model (response by randomly shuffling the references')
    parser.add_argument('-c', '--config')
    parser.add_argument('-p', '--prefix')
    args = parser.parse_args()
    prefix = Path(args.prefix or RANDOM_MODEL_ROOT)
    logging.basicConfig(level=logging.INFO)

    if args.config is None:
        from eval.repo import all_datasets as datasets
    else:
        datasets = load_config(args.config)['datasets']

    datasets = parse_dataset(datasets)
    # note the last line does not end with \n.
    for ds in datasets:
        logging.info('dataset: {}'.format(ds.references))
        references = Path(ds.references).read_text().splitlines()
        np.random.shuffle(references)

        output = get_model('random', ds.name).responses
        if not output.parent.exists():
            output.parent.mkdir(parents=True)
        output.write_text('\n'.join(references))
