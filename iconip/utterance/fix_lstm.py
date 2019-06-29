from iconip.utterance import load_feature, load_corr_matrix
import pandas as pd
import logging

TEST_KEY = ('lstm', 'lsdscc')
pd.set_option('display.max_columns', 100)


def load_and_print():
    test_value = load_feature()[TEST_KEY]
    print(test_value)


def show_corr_formula():
    from scipy.spatial.distance import pdist
    # 7. ``Y = pdist(X, 'correlation')``
    #
    #        Computes the correlation distance between vectors u and v. This is
    #
    #        .. math::
    #
    #           1 - \\frac{(u - \\bar{u}) \\cdot (v - \\bar{v})}
    #                    {{||(u - \\bar{u})||}_2 {||(v - \\bar{v})||}_2}


def show_nan_for_method(method):
    """
    Show whether the correlation calculated with method has any NaN.
    The result is for all 3 methods, there are NaNs.

    :param method:
    :return:
    """
    corr: pd.DataFrame = load_corr_matrix(method, *TEST_KEY)
    print(corr)


if __name__ == '__main__':
    # show_nan_for_method('spearman')
    logging.basicConfig(level=logging.INFO)
    show_nan_for_method('kendall')
    # TODO: we can turn NaN into 0.
