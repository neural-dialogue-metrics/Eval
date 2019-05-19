import logging
from pathlib import Path
from eval.data import find_all_data_files
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir_1')
    parser.add_argument('dir_2')
    args = parser.parse_args()

    data_files = find_all_data_files(args.dir_1)
    missing = 0
    inconsistent = 0

    for file in data_files:
        file_2 = Path(args.dir_2).with_name(file.name)
        if not file_2.is_file():
            missing += 1
            logging.info('dir_2: {} not a file'.format(file_2))
        elif file_2.read_bytes() != file.read_bytes():
            inconsistent += 1
            logging.info('file content not the same:')
            logging.info('file_1: {}'.format(file))
            logging.info('file_2: {}'.format(file_2))

    logging.info('missing {}'.format(missing))
    logging.info('inconsistent {}'.format(inconsistent))
