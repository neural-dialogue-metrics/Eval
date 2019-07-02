"""
Compute inter-metric correlation on example-level and serialize the data.

The data is then analyzed and visualized in various ways. See `corr_heatmap.py` and other scripts.

Methods are:
    - Pearson's r
    - Spearman's r
    - Kendall's tau
"""
from iconip import CORR_METHODS
from iconip.utterance import load_all_scores, SAVE_ROOT
from eval.utils import make_parent_dirs
import logging

logger = logging.getLogger(__name__)


def compute_corr():
    """
    Compute and serialize correlation matrix for all (instance, method)s.

    :return:
    """
    for key, value in load_all_scores().items():
        for method in CORR_METHODS:
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
