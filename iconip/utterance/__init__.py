import pickle
from pathlib import Path
from eval.data import UtterScoreDist, load_score_db_index
import pandas as pd
from eval.consts import DATA_V2_ROOT
import logging

logger = logging.getLogger(__name__)

SAVE_ROOT = Path(DATA_V2_ROOT) / 'iconip' / 'utterance'


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


def load_model_dataset2_feature(path=None):
    cache_path = SAVE_ROOT / 'feature' / 'dataframe_map.pkl'
    cache_path.parent.mkdir(exist_ok=True, parents=True)
    if cache_path.is_file():
        logging.info('loading features from cache file: {}'.format(cache_path))
        return pickle.load(cache_path.open('rb'))
    score_db_index = load_score_db_index(path)
    features = create_model_dataset2utter_feature(score_db_index)
    pickle.dump(features, cache_path.open('wb'))
    return features
