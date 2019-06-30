"""
Common code for example-level scores manipulation.
"""
import logging
import pickle
import pandas as pd

from pathlib import Path
from eval.consts import DATA_V2_ROOT
from eval.data import UtterScoreDist, load_score_db_index
from eval.normalize import normalize_name
from eval.utils import make_parent_dirs

logger = logging.getLogger(__name__)

# Namespace for generated data.
SAVE_ROOT = Path(DATA_V2_ROOT) / 'iconip' / 'utterance'

# Save the plots directly into paper's workspace.
# Plots change quite fast when fine-tuning.
PLOT_ROOT = Path('/home/cgsdfc/ICONIP2019/figure')


def make_score_matrix(score_index: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Create a score matrix from its index.

    :param score_index: a df for indcies to all metrics of an instance.
    :param kwargs: pass to `UtterScoreDist()`.
    :return: a `score_matrix`.
    """
    columns = {}
    for path in score_index.filename:
        utter = UtterScoreDist(path, normalize_names=True, **kwargs)
        columns[utter.metric] = utter.utterance
    # sort by columns names.
    return pd.DataFrame(columns).sort_index(axis=1)


def make_all_score_matrices(score_db_index, **kwargs):
    """
    Create a mapping from all instances to their score matrix.

    :param score_db_index: index to the score files returned by `load_score_db_index()`.
    :param kwargs: pass to the `UtterScoreDist()`.
    :return: a mapping of all score matrix.
    """
    result = {}
    for key, df in score_db_index.groupby(['model', 'dataset']):
        result[key] = make_score_matrix(df, **kwargs)
    return result


def load_all_scores(path=None, use_cache=True):
    """
    Load all scores with the same format as `make_all_score_matrices()` from `path` with caching.

    :param path: where to find the score files.
    :param use_cache: if true, a cached pickle file will be loaded instead of computing afresh.
    :return: see `make_all_score_matrices()`.
    """
    cache_path = SAVE_ROOT / 'feature' / 'dataframe_map.pkl'
    make_parent_dirs(cache_path)
    if cache_path.is_file() and use_cache:
        logging.info('loading score_data from cache file: {}'.format(cache_path))
        return pickle.load(cache_path.open('rb'))
    score_db_index = load_score_db_index(path)
    score_data = make_all_score_matrices(score_db_index)
    pickle.dump(score_data, cache_path.open('wb'))
    return score_data


def load_corr_matrix(method, model, dataset) -> pd.DataFrame:
    """
    Load the correlation matrix for a specific method and instance.

    :param method: a correlation method such as pearson. See `compute_corr.py`.
    :param model: the name of a model.
    :param dataset: the name of a dataset.
    :return: the correlation matrix
    """
    target = SAVE_ROOT / 'corr' / method / model / dataset / 'corr.json'
    logger.info('loading correlation of method {} for ({}, {})'.format(method, model, dataset))
    return pd.read_json(str(target))


def normalize_key(key):
    return map(normalize_name, ['model', 'dataset'], key)


def load_all_corr():
    """
    Load a mapping from all instances and all methods to their corresponding corr matrix.

    :return: a mapping from `(method, model, dataset)` to a df.
    """
    cache_path = SAVE_ROOT / 'corr' / 'all' / 'cache.pkl'
    if cache_path.is_file():
        return pickle.load(cache_path.open('rb'))

    root = SAVE_ROOT / 'corr'
    files = root.rglob('*.json')

    def path_to_items(path: Path):
        key = path.parts[-4:-1]
        return key, load_corr_matrix(*key)

    # Create the data afresh.
    all_corr = dict(map(path_to_items, files))
    make_parent_dirs(cache_path)
    pickle.dump(all_corr, cache_path.open('wb'))
    return all_corr


if __name__ == '__main__':
    # force recomputation.
    # load_feature(use_cache=False)
    print(load_all_corr())
