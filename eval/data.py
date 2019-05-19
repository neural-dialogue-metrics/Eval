import json
import logging
from pathlib import Path

import pandas as pd
import seaborn as sns
from eval.config_parser import product_models_datasets
from eval.consts import SAMPLE_SIZE, RANDOM_STATE, SEPARATOR, TRIPLE_NAMES, SCORE_DB_DIR, TIMES_NEW_ROMAN
from eval.normalize import normalize_name as _normalize_name
from pandas import DataFrame
from sklearn.preprocessing import scale as sklearn_scale

logger = logging.getLogger(__name__)


def get_model_dataset_pairs(models=None, datasets=None):
    from eval.repo import all_models, all_datasets

    if models is None:
        models = all_models
    if datasets is None:
        datasets = all_datasets

    return product_models_datasets(models, datasets)


def load_system_score(prefix: Path, remove_random_model=False, normalize_name=False):
    records = [json.load(file.open('r')) for file in prefix.rglob('*.json')]
    for data in records:
        del data['utterance']
    df = DataFrame.from_records(records)
    if remove_random_model:
        df = df[df.model != 'random'].reset_index()
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
            self._index = load_filename_data(self.data_dir)
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

    def __init__(self, filename: Path, scale=False, normalize=False):
        data = json.load(filename.open())
        super(UtterScoreDist, self).__init__(**data)
        self.system = data['system']
        self.scaled = scale
        self.normalized = normalize
        utterance = data['utterance']
        if scale:
            utterance = sklearn_scale(utterance)
        self.utterance = utterance
        if normalize:
            self.normalize_name_inplace()


def find_all_data_files(dir):
    return list(Path(dir).rglob('*.json'))


def remove_ppl_and_random(df: DataFrame):
    return df[(df.model != 'random') & (df.metric != 'serban_ppl')]


def load_filename_data(data_dir=None):
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
    logging.basicConfig(level=logging.INFO)
    sns.set(color_codes=True, font=TIMES_NEW_ROMAN)
