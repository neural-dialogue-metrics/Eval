import pickle
from pathlib import Path
from eval.data import UtterScoreDist, load_score_db_index
import pandas as pd
from eval.consts import DATA_V2_ROOT
from eval.normalize import normalize_name
import logging
from eval.utils import make_parent_dirs

logger = logging.getLogger(__name__)

SAVE_ROOT = Path(DATA_V2_ROOT) / 'iconip' / 'utterance'

PLOT_ROOT = Path('/home/cgsdfc/ICONIP2019/figure')


def create_utter_feature_df(score_index_per_model_dataset: pd.DataFrame, **kwargs):
    """
    Create a utterace-feature dataframe from an index to the actual scores per model-dataset.

    :param score_index_per_model_dataset:
    :param kwargs:
    :return:
    """
    columns = {}
    for path in score_index_per_model_dataset.filename:
        utter = UtterScoreDist(path, normalize_names=True, **kwargs)
        columns[utter.metric] = utter.utterance
    # sort by columns names.
    return pd.DataFrame(columns).sort_index(axis=1)


def create_model_dataset2utter_feature(score_db_index, **kwargs):
    """
    Create all utterance-feature dataframe for model & dataset combination.

    :param score_db_index: returned by load_score_db_index().
    :param kwargs: pass to the UtterScoreDist().
    :return:
    """
    result = {}
    for key, df in score_db_index.groupby(['model', 'dataset']):
        result[key] = create_utter_feature_df(df, **kwargs)
    return result


def load_feature(path=None, use_cache=True):
    cache_path = SAVE_ROOT / 'feature' / 'dataframe_map.pkl'
    make_parent_dirs(cache_path)
    if cache_path.is_file() and use_cache:
        logging.info('loading features from cache file: {}'.format(cache_path))
        return pickle.load(cache_path.open('rb'))
    score_db_index = load_score_db_index(path)
    features = create_model_dataset2utter_feature(score_db_index)
    pickle.dump(features, cache_path.open('wb'))
    return features


def load_corr_matrix(method, model, dataset):
    target = SAVE_ROOT / 'corr' / method / model / dataset / 'corr.json'
    logger.info('loading correlation of method {} for ({}, {})'.format(method, model, dataset))
    return pd.read_json(str(target))


def normalize_key(key):
    return map(normalize_name, ['model', 'dataset'], key)


def load_all_corr():
    cache_path = SAVE_ROOT / 'corr' / 'all' / 'cache.pkl'
    if cache_path.is_file():
        return pickle.load(cache_path.open('rb'))

    root = SAVE_ROOT / 'corr'
    files = root.rglob('*.json')

    def path_to_items(path: Path):
        key = path.parts[-4:-1]
        return key, load_corr_matrix(*key)

    all_corr = dict(map(path_to_items, files))
    make_parent_dirs(cache_path)
    pickle.dump(all_corr, cache_path.open('wb'))
    return all_corr


if __name__ == '__main__':
    # force recomputation.
    # load_feature(use_cache=False)
    print(load_all_corr())
