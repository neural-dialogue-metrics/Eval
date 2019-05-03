from eval.utils import Dataset
from eval.config import models, datasets
from eval.consts import CONTEXTS, REFERENCES


def get_model(name, trained_on):
    for model in models:
        if name == model.name and trained_on == model.trained_on:
            return model
    raise ValueError('unknown model: {} on {}'.format(name, trained_on))


def get_dataset(name):
    try:
        params = datasets[name]
    except KeyError as e:
        raise ValueError('unknown dataset: {}'.format(name)) from e
    return Dataset(
        name=name,
        references=params[REFERENCES],
        contexts=params[CONTEXTS],
    )
