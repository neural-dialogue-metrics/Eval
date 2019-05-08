from pathlib import Path
import logging
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from corr.consts import *
from corr.utils import DataIndex
from corr.utils import remake_needed

NAME = 'pairplot'

GROUP_MAP = {
    'word_overlap': [
        'bleu_4',
        'rouge_2',
        'meteor',
    ],
    'rouge': 'all',
    'bleu': 'all',
    'distinct_n': 'all',
    'embedding_based': 'all',
}

logger = logging.getLogger(__name__)


# pairplot/<ds>/model/<model>/group/
def get_output(prefix: Path, dataset, mode, mode_arg, group):
    return prefix / NAME / dataset / mode / mode_arg / group / PLOT_FILENAME


def do_plot(list_of_scores, mode, output, group_id):
    x = list_of_scores[0]
    if mode == MODE_METRIC:
        df = pd.DataFrame({
            item.model: item.utterance for item in list_of_scores,
        })
        pair_grid = sns.pairplot(
            data=df,
            hue='model',
            kind='reg',
        )
        pair_grid.fig.suptitle('{} of {} models on {}'.format(x.metric, group_id, x.dataset))
    else:
        df = pd.DataFrame({
            item.metric: item.utterance for item in list_of_scores
        })
        pair_grid = sns.pairplot(
            data=df,
            hue='metric',
            kind='reg',
        )
        pair_grid.fig.suptitle('{} metrics of {} on {}'.format(group_id, x.model, x.dataset))
    logger.info('plotting to {}'.format(output))
    pair_grid.savefig(str(output))
    plt.close()


def get_and_scale_data(data_index, df):
    return [
        data_index.get_data(row.filename, scale=True)
        for row in df.itertuples(index=False, name='Triple')
    ]


def get_sources(df: pd.DataFrame):
    return df['filename'].values


def get_metric_group(df: pd.DataFrame, group_name, rules: Union[list, str]):
    if rules == 'all':
        pattern = group_name
    else:
        pattern = '|'.join(map(lambda r: '(' + r + ')', rules))
    logger.info('pattern {}'.format(pattern))
    return df[df['metric'].str.contains(pattern, regex=True)]


def plot(data_index: DataIndex, prefix: Path, group_map=None, force=False):
    def plot_mode_metric():
        logger.info('plotting for mode {}'.format(MODE_METRIC))
        group_name = 'all'
        for (dataset, metric), data in data_index.index.groupby(['dataset', 'metric']):
            target = get_output(prefix, dataset, MODE_METRIC, metric, group_name)
            sources = get_sources(data)
            if remake_needed(target, *sources, force=force):
                target.parent.mkdir(parents=True, exist_ok=True)
                real_data = get_and_scale_data(data_index, data)
                do_plot(real_data, MODE_METRIC, target, group_name)

    def plot_mode_model():
        logger.info('plotting mode {}'.format(MODE_MODEL))
        for (dataset, model), data in data_index.index.groupby(['dataset', 'model']):
            for group_name, rules in group_map.items():
                group_df = get_metric_group(data, group_name, rules)
                target = get_output(prefix, dataset, MODE_MODEL, model, group_name)
                sources = get_sources(group_df)
                if remake_needed(target, *sources, force=force):
                    target.parent.mkdir(parents=True, exist_ok=True)
                    real_data = get_and_scale_data(data_index, group_df)
                    do_plot(real_data, MODE_MODEL, target, group_name)

    if group_map is None:
        group_map = GROUP_MAP

    plot_mode_metric()
    plot_mode_model()


if __name__ == '__main__':
    from corr.utils import plot_main

    plot_main()
