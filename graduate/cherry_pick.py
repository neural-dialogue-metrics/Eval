import logging
from pathlib import Path

import pandas as pd
from graduate.annotated import load_annotated_index, FILENAME
from eval.consts import DATA_V2_ROOT
from eval.utils import make_parent_dirs
from pandas import DataFrame, Series

__version__ = '0.0.1'

NAMESPACE = 'cherry_pick'

CRR = ['context', 'reference', 'response']


def get_output(prefix: Path, k, dataset, model, metric):
    return make_parent_dirs(prefix / 'example' / NAMESPACE / str(k) / dataset / model / metric / FILENAME)


def get_metrics(df: DataFrame):
    for col, dtype in df.dtypes.items():
        if col not in CRR:
            yield col


def make_cherry_pick(df: DataFrame, k=None, prefix: Path = None):
    if k is None:
        k = 1
    if prefix is None:
        prefix = Path(DATA_V2_ROOT)

    for row in df.itertuples(index=False):
        df2: DataFrame = pd.read_json(row.path)
        metrics = list(get_metrics(df2))
        for metric in metrics:
            df3 = df2.nlargest(n=k, columns=metric)
            crr: Series = df3.iloc[0][CRR]
            crr['score'] = df3[metric].iloc[0]
            output = get_output(
                prefix=prefix,
                k=k,
                dataset=row.dataset,
                model=row.model,
                metric=metric,
            )
            logging.info('writing to {}'.format(output))
            crr.to_json(output)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    df = load_annotated_index()
    make_cherry_pick(df)
