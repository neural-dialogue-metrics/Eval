"""
Generate fully-annotated examples for each dataset.

An annotated example is a self-contained record with everything you need:
    - the context.
    - the generated responses.
    - the reference (ground-truth).
    - the scores of all metrics.

Each test set defines a collection of annotated examples, which is stored per instance.
With that you can easily compare the scores of different metrics and relate the differences to the
actual dialogue being scored although the analysis is generally hard.
"""
import logging
from pathlib import Path

from pandas import DataFrame, Series

from eval.consts import DATA_V2_ROOT
from eval.consts import REFERENCES, CONTEXTS, RESPONSES
from eval.data import UtterScoreDist
from eval.data import load_score_db_index
from eval.utils import make_parent_dirs

NAMESPACE = 'annotated'

FILENAME = 'chat.json'

DATASET_FILES = ('contexts', 'references')

__version__ = '0.0.2'


def get_output(prefix: Path, dataset: str, model: str):
    return make_parent_dirs(prefix / NAMESPACE / dataset / model / FILENAME)


def get_the_file(dir: Path, subdir: str):
    """
    Extract the specific file under `dir/subdir`.

    There should be exactly one file there, as the dirs are only namespaces.

    :param dir: the dir for an object, such as a model or dataset.
    :param subdir: the dir for an attribute of that object, such as the response for a model or
                    the context for a dataset.
    :return: the path of `dir/subdir/the_file`, or conceptually, the value of `object.attr`.
    """
    files = list(dir.joinpath(subdir).glob('*'))
    if len(files) != 1:
        raise ValueError('there should be exactly one file')
    return files[0]


def load_examples_index(prefix: Path = None, remove_random_model=False) -> DataFrame:
    """
    Load an index of examples from a predefined directory structure.

    This is similar to run the command `find` to get a list of paths and split each path into components.
    The components and the path itself then become one of the records in the resultant df.
    It is called an index since it holds the _path_ to the actual data (not the data).

    :param prefix: dir to look for example files.
    :param remove_random_model: if true, remove everything of the random model.
    :return: the index df.
    """
    if prefix is None:
        prefix = Path('/home/cgsdfc/Metrics/Eval/data/v2/example/all')

    def walk_dir():
        """
        Walk `prefix` and yield dicts as records.
        It defines the columns of the returned df:
            - dataset
            - model
            - context_file
            - reference_file
            - response_file

        :return:
        """
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
    """
    Create all the annotated examples and serialize them to disk.

    :param df: the index created by `make_annotated_example_index()`.
    :param prefix: the dir to save data to.
    :return:
    """
    for (dataset, model), df2 in df.groupby(['dataset', 'model']):
        # Extract the file contents of the three files.
        series: Series = df2.iloc[0][['context_file', 'reference_file', 'response_file']]
        series = series.apply(lambda file: Path(file).read_text().splitlines())
        columns = series.to_dict(dict)
        # Add scores of all metrics to it.
        for row in df2.itertuples(index=False):
            score = UtterScoreDist(row.score_file)  # load score
            columns[row.metric] = score.utterance
        # Strip the `_file` suffix as it is no longer true.
        df3 = DataFrame(data=columns).rename(columns=lambda str: str.replace('_file', ''))
        output = get_output(prefix, dataset, model)
        logging.info('writing to {}'.format(output))
        # Save as `records` format for human-readability of the json files.
        # One json object for one example.
        df3.to_json(path_or_buf=output, orient='records')


def make_annotated_example_index():
    """
    Make an index of the annotated examples.

    An annotated example consists of two parts, the example and its scores on various metrics.
    The two parts are stored separately and our task is to join them together.
    `load_examples_index()` and `load_score_db_index()` load the indices for the examples and their scores
    respectively. This function joins them and renames the column names.

    The returned df has these columns:
        - model, dataset: instance identifier.
        - score_file, context_file, response_file, reference_file: path to various files.

    :return: a df for the index for the annotated examples.
    """
    df1 = load_examples_index(remove_random_model=True)
    df2 = load_score_db_index()
    # df.join(x) requires x has an index to perform the join on.
    # We join on the model instance identifier (model, dataset) since it is the primary key.
    df3 = df1.join(
        df2.rename(columns={'filename': 'score_file'}).set_index(['model', 'dataset']),
        on=['model', 'dataset']
    )
    df3 = df3.sort_values(['dataset', 'model', 'metric'])
    return df3.reset_index()


def load_annotated_index(prefix: Path = None):
    """
    Load an index of the annotated_index from a predefined directory structure.

    Unlike `make_annotated_example_index()`, which creates the index from stretches, this function
    is an interface to the annotated examples that have been serialized to the disk. It makes it easy to
    access these examples with a `pandas.DataFrame`.

    :param prefix: dir to annotated examples.
    :return: a df as the index.
    """
    if prefix is None:
        prefix = Path(DATA_V2_ROOT) / 'example' / NAMESPACE
    files = list(prefix.rglob('*.json'))

    def parse(path: Path):
        dataset, model = path.parts[-3:-1]
        return locals()

    return DataFrame.from_records(map(parse, files))


if __name__ == '__main__':
    # Generate these examples.
    logging.basicConfig(level=logging.INFO)
    df = make_annotated_example_index()
    make_annotated(df, prefix=Path(DATA_V2_ROOT).joinpath('example'))
