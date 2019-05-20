from pathlib import Path

from eval.data import UtterScoreDist
from eval.consts import REFERENCES, CONTEXTS, RESPONSES
from eval.data import load_score_db_index
from pandas import DataFrame, Series

from eval.utils import make_parent_dirs

NAMESPACE = 'annotated'

FILENAME = 'chat.json'

DATASET_FILES = ('contexts', 'references')

__version__ = '0.0.1'


def get_output(prefix: Path, dataset: str, model: str):
    return make_parent_dirs(prefix / NAMESPACE / dataset / model / FILENAME)


def get_the_file(dir: Path, subdir: str):
    files = list(dir.joinpath(subdir).glob('*'))
    if len(files) != 1:
        raise ValueError('there should be exactly one file')
    return files[0]


def load_examples_index(prefix: Path = None, remove_random_model=False) -> DataFrame:
    if prefix is None:
        prefix = Path('/home/cgsdfc/Metrics/Eval/data/v2/example/all')

    def walk_dir():
        for dataset_dir in prefix.iterdir():
            for model_dir in dataset_dir.joinpath('model').iterdir():
                record = {
                    'dataset': dataset_dir.name,
                    'model': model_dir.name,
                    'context_file': get_the_file(dataset_dir, CONTEXTS),
                    'reference_file': get_the_file(dataset_dir, REFERENCES),
                    'response_file': get_the_file(model_dir, RESPONSES),
                }
                yield record

    df = DataFrame.from_records(walk_dir())
    if remove_random_model:
        df = df[df.model != 'random']
    return df


def make_annotated(df: DataFrame):
    for (dataset, model), df2 in df.groupby(['dataset', 'model']):
        series: Series = df2.iloc[0][['context_file', 'reference_file', 'response_file']]
        series = series.apply(lambda file: Path(file).read_text().splitlines())
        columns = series.to_dict(dict)
        for row in df2.itertuples(index=False):
            score = UtterScoreDist(row.score_file)
            columns[row.metric] = score.utterance
        df3 = DataFrame(data=columns).rename(columns=lambda str: str.replace('_file', ''))
        print(df3)


def load_example_score_index():
    df1 = load_examples_index(remove_random_model=True)
    df2 = load_score_db_index()
    df3 = df1.join(
        df2.rename(columns={'filename': 'score_file'}).set_index(['model', 'dataset']),
        on=['model', 'dataset']
    )
    df3 = df3.sort_values(['dataset', 'model', 'metric'])
    return df3.reset_index()


if __name__ == '__main__':
    df = load_example_score_index()
    make_annotated(df)
