from eval.data import UtterScoreDist, load_score_db_index
import pandas as pd


def create_utter_feature_df(score_index_per_model_dataset: pd.DataFrame, **kwargs):
    """
    Create a utterace-feature dataframe from an index to the actual scores per model-dataset.

    :param score_index_per_model_dataset:
    :param kwargs:
    :return:
    """
    columns = {}
    for path in score_index_per_model_dataset.filename:
        utter = UtterScoreDist(path, **kwargs)
        columns[utter.metric] = utter.utterance
    return pd.DataFrame(columns)


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


if __name__ == '__main__':
    score_db_index = load_score_db_index()
    create_model_dataset2utter_feature(score_db_index)
