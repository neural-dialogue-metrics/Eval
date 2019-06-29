import json
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
import seaborn as sns
from eval.config_parser import product_models_datasets
from eval.consts import SAMPLE_SIZE, RANDOM_STATE, SEPARATOR, TRIPLE_NAMES, SCORE_DB_DIR, TIMES_NEW_ROMAN
from eval.normalize import normalize_name as _normalize_name
from pandas import DataFrame
from sklearn.preprocessing import scale as sklearn_scale

logger = logging.getLogger(__name__)


def get_model_dataset_pairs(models=None, datasets=None):
    """
    Compute the product of a list of models and a list of datasets.

    :param models:
    :param datasets:
    :return:
    """
    from eval.repo import all_models, all_datasets

    if models is None:
        models = all_models
    if datasets is None:
        datasets = all_datasets

    return product_models_datasets(models, datasets)


default_removes = (
    ('model', 'random'),
    ('metric', 'serban_ppl',)
)


def _remove_columns(df: DataFrame, remove: Tuple[str, str]):
    for col_name, col_val in remove:
        df = df[df[col_name] != col_val]
    return df


def load_system_score(prefix: Path = None,
                      normalize_name=True,
                      remove_random_model=True,
                      remove_serban_ppl=True):
    """
    Load the system scores for all settings from json files.
    The returned DataFrame has the following columns:
        - metric: the name of the metric
        - model: the name of the model
        - dataset: the name of the dataset
        - system: the value of the system score.

    :param prefix:
    :param normalize_name: if true, all names are normalized as appearing in the paper: bleu_1 becomes BLEU-1.
    :param remove_random_model: if true, all the rows of the random model is removed.
    :param remove_serban_ppl: if true, all the rows whose metric is serban_ppl is removed.
    :return:
    """
    if prefix is None:
        prefix = Path(SCORE_DB_DIR)
    records = [json.load(file.open('r')) for file in prefix.rglob('*.json')]
    for data in records:
        del data['utterance']
    df = DataFrame.from_records(records)
    if remove_random_model:
        df = df[df.model != 'random'].reset_index()
    if remove_serban_ppl:
        df = df[df.metric != 'serban_ppl'].reset_index()
    if normalize_name:
        all_cols = ['model', 'dataset', 'metric']
        for col in all_cols:
            normalized = df[col].apply(lambda x: _normalize_name(col, x))
            df[col] = normalized
    return df


def scale_and_sample(frame: pd.DataFrame):
    return frame.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).transform(sklearn_scale)


class DataIndex:

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir).absolute()
        self._index = None
        self._cache = {}

    @property
    def index(self):
        if self._index is None:
            self._index = load_score_db_index(self.data_dir)
        return self._index

    def iter_triples(self):
        return self.index.itertuples(index=False, name='Triple')

    def get_data(self, path, **kwargs):
        if path in self._cache:
            return self._cache[path]
        return self._cache.setdefault(path, UtterScoreDist(path, **kwargs))


class Triple:
    def __init__(self, model, dataset, metric, **kwargs):
        self.model = model
        self.dataset = dataset
        self.metric = metric

    @property
    def parts(self):
        return self.model, self.dataset, self.metric

    @property
    def name(self):
        return SEPARATOR.join((self.model, self.dataset, self.metric))

    def normalize_name_inplace(self):
        for name in TRIPLE_NAMES:
            normalized = _normalize_name(name, getattr(self, name))
            setattr(self, name, normalized)


class UtterScoreDist(Triple):
    """Utterance-Score Distribution"""

    def __init__(self, filename: Path, scale_values=False, normalize_names=False):
        """
        Create an UtterScoreDist.
        
        :param filename: the json file to look for data. 
        :param scale_values: if true, the values will be scaled to have mean=0 and std=1.
        :param normalize_names: if true, the names will be normalized.
        """
        data = json.load(filename.open())
        super(UtterScoreDist, self).__init__(**data)
        self.system = data['system']
        self.scaled = scale_values
        self.normalized = normalize_names
        utterance = data['utterance']
        if scale_values:
            utterance = sklearn_scale(utterance)
        self.utterance = utterance
        if normalize_names:
            self.normalize_name_inplace()


def find_all_data_files(dir):
    return list(Path(dir).rglob('*.json'))


def remove_ppl_and_random(df: DataFrame):
    return df[(df.model != 'random') & (df.metric != 'serban_ppl')]


def load_score_db_index(data_dir=None):
    """
    Load an index of the score database. The real score data are not loaded.
    Only the (metric, model, dataset) and the path to the real data are loaded.
    The resultant DataFrame has these columns:
        - metric
        - model
        - dataset
        - filename

    :param data_dir:
    :return:
    """
    if data_dir is None:
        data_dir = SCORE_DB_DIR
    logger.info('loading filename data from {}'.format(data_dir))
    data_files = find_all_data_files(data_dir)

    def parse(filename: Path):
        metric, model, dataset = filename.parent.parts[-1:-4:-1]
        return locals()

    df = DataFrame.from_records([parse(p) for p in data_files])
    return remove_ppl_and_random(df)


def remake_needed(target: Path, *sources, force=False):
    if force:
        return True
    if not target.exists():
        return True
    for src in sources:
        if not src.exists():
            raise ValueError('cannot remake with {}'.format(src))
        if target.stat().st_mtime < src.stat().st_mtime:
            return True
    logger.info('{} is up to date'.format(target))
    return False


def seaborn_setup():
    sns.set(color_codes=True, font=TIMES_NEW_ROMAN)


def get_schema_name(__file__):
    return Path(__file__).stem
