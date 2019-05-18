import json
from pathlib import Path

from pandas import DataFrame

from eval.config_parser import product_models_datasets

from corr.normalize import normalize_name as _normalize_name


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
        df = df[df.model != 'random']
    if normalize_name:
        all_cols = ['model', 'dataset', 'metric']
        for col in all_cols:
            normalized = df[col].apply(lambda x: _normalize_name(col, x))
            df[col] = normalized
    return df
