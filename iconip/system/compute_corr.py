"""
Compute inter-instance system-level correlatoin
"""
import logging
from eval.utils import make_parent_dirs
from iconip.system import make_system_scores, SAVE_ROOT
from iconip import CORR_METHODS


def compute():
    scores = make_system_scores()
    scores = scores.rename(index=lambda x: '({}, {})'.format(*x))  # Make tuple columns to str.
    print(scores.index)
    scores = scores.transpose()
    for method in CORR_METHODS:
        logging.info('computing method {}'.format(method))
        corr = scores.corr(method)
        output = make_parent_dirs(SAVE_ROOT / 'corr' / method / 'corr.json')
        corr.to_json(output)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    compute()
