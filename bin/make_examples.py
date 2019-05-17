import argparse
import logging
import re
import shutil
from operator import attrgetter
from pathlib import Path

from eval.consts import *
from eval.utils import subdirs
from pandas import DataFrame

logger = logging.getLogger(__name__)


# Copy files of CRR to a centralized location.
def localize_files(prefix: Path):
    from eval.repo import all_models, all_datasets

    def localize_models():
        for model in all_models:
            path = prefix / model.trained_on / 'model' / model.name / RESPONSES
            path.mkdir(parents=True, exist_ok=True)
            shutil.copy(model.responses, path)

    def localize_datasets():
        for dataset in all_datasets:
            path = prefix / dataset.name

            def do_copy(name):
                src = getattr(dataset, name)
                dst = path / name
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dst)

            do_copy(CONTEXTS)
            do_copy(REFERENCES)

    localize_models()
    localize_datasets()


def print_examples(prefix: Path, n=None, eot_token=None, seed=None, mode=None):
    if n is None: n = 6
    if seed is None: seed = 666
    if eot_token is None: eot_token = '->'
    if mode is None:
        mode = 'latex'

    def cleanup(utter):
        utter = (utter
                 .replace('</s>', ' ')
                 .replace('__eou__', ' ')
                 .replace('__eot__', eot_token))
        utter = re.sub(r'\s+', ' ', utter)
        return utter

    def get_file(path: Path):
        under = list(path.glob('*'))[0]
        return under.read_text().splitlines()

    def iter_sessions():
        for ds in subdirs(prefix):
            session = {
                CONTEXTS: get_file(ds / CONTEXTS),
                REFERENCES: get_file(ds / REFERENCES),
            }
            model_dir = ds / 'model'
            for model in subdirs(model_dir):
                session[model.name] = get_file(model / RESPONSES)
            yield ds.name, session

    ignored = ('index', 'random')

    def print_plain():
        for name, sess in iter_sessions():
            data: DataFrame = DataFrame(sess).sample(n=n, random_state=seed).reset_index()
            print('Dataset: {}'.format(name))
            print('===================')
            for index, row in data.iterrows():
                print('Example #{}'.format(index))
                for index, value in row.iteritems():
                    if index in ignored:
                        continue
                    print('{}: {}'.format(index.capitalize(), value))
                print()
            print()

    def print_latex_table():
        order = (CONTEXTS, REFERENCES, 'lstm', 'vhred', 'hred')
        for name, sess in iter_sessions():
            data: DataFrame = DataFrame(sess).sample(n=n, random_state=seed).reset_index()
            print('Dataset: {}'.format(name))
            for index, row in data.iterrows():
                line = ' & '.join([getattr(row, attr) for attr in order])
                print(line, end=' \\\\\n')
                print('\\hline')
            print()

    if mode == 'latex':
        print_latex_table()
    elif mode == 'plain':
        print_plain()
    else:
        raise ValueError


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Make examples by randomly sampling')
    parser.add_argument('-p', '--prefix', help='output prefix')
    parser.add_argument('-n', '--n-examples', type=int, default=None, help='number of examples to draw')
    parser.add_argument('-l', '--localize', action='store_true', help='localize the files')
    parser.add_argument('-s', '--seed', type=int, default=None, help='random seed')
    parser.add_argument('-x', '--print-examples', action='store_true')
    args = parser.parse_args()

    if args.localize:
        localize_files(Path(args.prefix))
    elif args.print_examples:
        print_examples(
            prefix=Path(args.prefix),
            n=args.n_examples,
            seed=args.seed,
        )
