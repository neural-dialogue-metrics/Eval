import logging
from pathlib import Path

from eval.data import UtterScoreDist
from eval.consts import REFERENCES, CONTEXTS, RESPONSES
from eval.consts import DATA_V2_ROOT

from eval.data import load_score_db_index
from pandas import DataFrame, Series

from eval.utils import make_parent_dirs

NAMESPACE = 'annotated'

FILENAME = 'chat.json'

DATASET_FILES = ('contexts', 'references')

__version__ = '0.0.2'


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


def make_annotated(df: DataFrame, prefix: Path):
    for (dataset, model), df2 in df.groupby(['dataset', 'model']):
        series: Series = df2.iloc[0][['context_file', 'reference_file', 'response_file']]
        series = series.apply(lambda file: Path(file).read_text().splitlines())
        columns = series.to_dict(dict)
        for row in df2.itertuples(index=False):
            score = UtterScoreDist(row.score_file)
            columns[row.metric] = score.utterance
        df3 = DataFrame(data=columns).rename(columns=lambda str: str.replace('_file', ''))
        output = get_output(prefix, dataset, model)
        logging.info('writing to {}'.format(output))
        df3.to_json(path_or_buf=output, orient='records')


def load_example_score_index():
    df1 = load_examples_index(remove_random_model=True)
    df2 = load_score_db_index()
    df3 = df1.join(
        df2.rename(columns={'filename': 'score_file'}).set_index(['model', 'dataset']),
        on=['model', 'dataset']
    )
    df3 = df3.sort_values(['dataset', 'model', 'metric'])
    return df3.reset_index()


def load_annotated_index(prefix: Path = None):
    if prefix is None:
        prefix = Path(DATA_V2_ROOT) / 'example' / NAMESPACE
    files = list(prefix.rglob('*.json'))

    def parse(path: Path):
        dataset, model = path.parts[-3:-1]
        return locals()

    return DataFrame.from_records(map(parse, files))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    df = load_example_score_index()
    make_annotated(df, prefix=Path(DATA_V2_ROOT).joinpath('example'))
