"""
Compute inter-metric correlation on example-level and serialize the data.

Methods are:
    - Pearson's r
    - Spearman's r
    - Kendall's tau
"""
from iconip.utterance import load_feature, SAVE_ROOT
from eval.utils import make_parent_dirs
import logging

logger = logging.getLogger(__name__)


def compute_corr():
    """
    Compute and serialize correlation matrix for all (instance, method)s.

    :return:
    """
    for key, value in load_feature().items():
        for method in ['pearson', 'spearman', 'kendall']:
            logger.info('computing ({}, {}, {})'.format(method, *key))
            output = make_parent_dirs(
                SAVE_ROOT / 'corr' / method / key[0] / key[1] / 'corr.json'
            )
            corr = value.corr(method=method)
            # Note: NaN causes the linkage() to raise!
            # corr(X, Y) is NaN when one of X, Y is all-zero.
            corr.fillna(0, inplace=True)
            corr.to_json(output)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    compute_corr()
