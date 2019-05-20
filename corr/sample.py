import logging
from pathlib import Path

import pandas as pd
from corr.annotated import load_annotated_index, FILENAME
from eval.consts import DATA_V2_ROOT
from eval.utils import make_parent_dirs
from pandas import DataFrame, Series

__version__ = '0.0.1'

NAMESPACE = 'sample'

CRR = ['context', 'reference', 'response']


def get_output(prefix: Path, k, dataset, model):
    return make_parent_dirs(prefix / 'example' / NAMESPACE / str(k) / dataset / model / FILENAME)


def get_metrics(df: DataFrame):
    for col, dtype in df.dtypes.items():
        if col not in CRR:
            yield col


def make_sample(df: DataFrame, k=None, prefix: Path = None):
    if k is None:
        k = 10
    if prefix is None:
        prefix = Path(DATA_V2_ROOT)

    for row in df.itertuples(index=False):
        df2: DataFrame = pd.read_json(row.path)
        df2 = df2.sample(n=k)
        crr = df2[CRR]
        output = get_output(
            prefix=prefix,
            k=k,
            dataset=row.dataset,
            model=row.model,
        )
        logging.info('writing to {}'.format(output))
        crr.to_json(output, orient='records')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    df = load_annotated_index()
    make_sample(df)
