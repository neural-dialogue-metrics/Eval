from eval.config_parser import product_models_datasets
from eval.repo import all_models, all_datasets


def get_model_dataset_pairs(models=None, datasets=None):
    if models is None:
        models = all_models
    if datasets is None:
        datasets = all_datasets

    return product_models_datasets(models, datasets)
