"""
Temp script used to fix the LSTM problems.
"""
from iconip.utterance import load_all_scores, load_corr_matrix
import pandas as pd
import logging
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, spearmanr, pearsonr
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.sparse import coo_matrix

TEST_KEY = ('lstm', 'lsdscc')
pd.set_option('display.max_columns', 100)


def load_and_print():
    test_value = load_all_scores()[TEST_KEY]
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


def linkage_precomputed_corr():
    """
    Use precomputed correlation matrix as the distance matrix for linkage.
    Perform necessary transformations. Correlation measures the degree of similarity, while
    distance measures the degree of dissimilarity.
    We make $dist = 1 - corr$.
    See this post:
    https://stackoverflow.com/questions/38070478/how-to-do-clustering-using-the-matrix-of-correlation-coefficients

    :return:
    """

    # optimal_ordering : bool, optional
    #         If True, the linkage matrix will be reordered so that the distance
    #         between successive leaves is minimal. This results in a more intuitive
    #         tree structure when the data are visualized. defaults to False, because
    #         this algorithm can be slow, particularly on large datasets [2]_. See
    #         also the `optimal_leaf_ordering` function.
    corr: pd.DataFrame = load_corr_matrix('pearson', *TEST_KEY)
    condensed = squareform(corr, checks=False, force='tovector')
    condensed = 1 - condensed
    Z = linkage(condensed, method='average', optimal_ordering=True)
    dendrogram(Z, orientation='left', leaf_label_func=lambda x: corr.columns[x])
    plt.show()


def convert_dense_matrix_to_condensed():
    """
    Since linkage() requires a condense distance matrix as returned by pdist()
    and we have a dense matrix computed with DataFrame.corr(), we need a way to convert the matrix to vector.
    So our pre-computed NaN-free correlation can serve as the distance matrix.
    The squareform() function does just that.

    :return:
    """
    corr: pd.DataFrame = load_corr_matrix('pearson', *TEST_KEY)
    print(squareform(corr, checks=False, force='tovector'))


if __name__ == '__main__':
    linkage_precomputed_corr()
