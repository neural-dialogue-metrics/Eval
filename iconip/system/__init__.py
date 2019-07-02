"""
Code for analysis of the system scores.
"""

from eval.data import load_system_score
from pandas import DataFrame
from iconip import cache_this, SAVE_ROOT


@cache_this(SAVE_ROOT / 'score' / 'cache.pkl')
def make_system_scores() -> DataFrame:
    """
    Create a df of (N, M) where N is the number of instances and M is the number of metrics.

    The `long_form_scores` returned by `load_system_score()` consists of (instance, score) records.
    With transformation it becomes records fields of instances. The fields are actually scores of
    various metrics. In this way, the df defines a matrix suitable for the analysis of
    pairwise correlation.

    :param long_form_scores: long-form systems scores by `load_system_score()`.
    :return: a matrix of `(#instances, #metrics)`.
    """
    long_form_scores: DataFrame = load_system_score()
    indices = []
    records = []
    for instance, value in long_form_scores.groupby(['model', 'dataset']):
        indices.append(instance)
        record = dict(value[['metric', 'system']].values)
        records.append(record)
    return DataFrame.from_records(records, indices)


if __name__ == '__main__':
    df = make_system_scores()
    print(df)
