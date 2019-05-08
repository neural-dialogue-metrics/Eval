import logging
import pickle

from eval.config_parser import parse_dataset
from eval.repo import all_datasets

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    datasets = parse_dataset(all_datasets)
    for ds in datasets:
        logging.info('checking {}'.format(ds))
        len_contexts = len(open(ds.contexts).readlines())
        len_references = len(open(ds.references).readlines())
        len_test_dialog = len(pickle.load(open(ds.test_dialogues, 'rb')))

        if len_contexts != len_references:
            logging.warning('{}: contexts != reference'.format(ds))
        if len_references != len_test_dialog:
            logging.warning('{}: references != test_dialog'.format(ds))
