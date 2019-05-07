import re
from pathlib import Path

import json
import seaborn as sns
from corr.utils import UtterScoreDist, load_filename_data
import matplotlib.pyplot as plt

GROUPBY_ALL = ('dataset', 'model', 'metric')
GROUPBY_MODEL = ('dataset', 'model')
GROUPBY_METRIC = ('dataset', 'metric')

config = {
    'distplot': {
        'all': True,
    }
}

url_map = {
    'distplot': [
        # distribution of (model, dataset, metric).
        {
            'url': '<plot>/<dataset>/<model>/<metric>/',
            'groupby': GROUPBY_ALL,
        }
    ],
    'jointplot': [
        # metric-to-metric scatter on (model, dataset).
        {
            'url': '<plot>/<dataset>/model/<model>/<metric_1>-<metric_2>/',
            'groupby': GROUPBY_MODEL,
        },

        # model-to-model scatter on (metric, dataset).
        {
            'url': '<plot>/<dataset>/metric/<metric>/<model_1>-<model_2>/',
            'groupby': GROUPBY_METRIC,
        },
    ],
    'pairplot': [
        {
            'url': '<plot>/<dataset>/model/<model>/all/',
            'groupby': GROUPBY_MODEL,
        },
        {
            'url': '<plot>/<dataset>/metric/<metric>/all/',
            'groupby': GROUPBY_METRIC,
        }

    ]
}


def materialize_url(url: str, **kwargs):
    for name, value in kwargs.items():
        url = url.replace('<{}>'.format(name), value)
    return url


def joint_plot(score_data, url, output_dir: Path, params: dict):
    pass


plot_fns = {
    'distplot': dist_plot,

}
