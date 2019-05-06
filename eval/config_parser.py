import itertools
import json
import logging
from pathlib import Path

from eval.repo import model_path
from eval.utils import Model, Dataset

logger = logging.getLogger(__name__)


def parse_metrics(config):
    try:
        from eval.metrics import metrics_classes
    except ImportError:
        logger.error('metric_classes not available. Some of the packages were not installed?')
        raise

    metrics = []
    if isinstance(config, list):
        for m in config:
            if isinstance(m, dict):
                name = m['name']
                cls = metrics_classes[name]
                metrics.extend(cls.parse_config(m))
            else:
                # assume to be MetricWrapper
                metrics.append(m)
    elif isinstance(config, dict):
        for name, metric_config in config.items():
            cls = metrics_classes[name]
            metrics.extend(cls.parse_config(metric_config))
    else:
        raise TypeError('metric config must be a list or a dict')
    return metrics


def parse_dataset(config):
    if isinstance(config, list):
        if not all(isinstance(ds, Dataset) for ds in config):
            raise TypeError('if dataset config is a list, it must contain only Dataset')
        return config
    dataset = []
    for name, value in config.items():
        dataset.append(Dataset(name=name, **value))
    return dataset


def parse_models(config):
    models = []
    for data_path in config:
        if isinstance(data_path, Model):
            model = data_path
        elif isinstance(data_path, str):
            model = model_path(data_path)
        else:
            raise TypeError('model config must be a list of str or Model')
        models.append(model)
    return models


def parse_models_and_datasets(config):
    models = parse_models(config.get('models'))
    datasets = parse_dataset(config.get('datasets'))
    return product_models_datasets(datasets, models)


def product_models_datasets(datasets, models):
    return [
        (model, dataset) for model, dataset in itertools.product(models, datasets)
        if model.trained_on == dataset.name
    ]


def load_config(filename):
    filename = Path(filename)
    if filename.suffix == '.json':
        config = json.load(filename.open())
    elif filename.suffix == '.py':
        code = compile(filename.read_text(), filename, 'exec')
        globals = {}
        exec(code, globals)
        config = globals['config']
    else:
        raise ValueError('invalid file type for config file: {}'.format(filename))
    return config


class BasicPayload:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    @property
    def context_file(self):
        return self.dataset.contexts

    @property
    def reference_file(self):
        return self.dataset.references

    @property
    def response_file(self):
        return self.model.responses

    @property
    def model_name(self):
        return self.model.name

    @property
    def dataset_name(self):
        return self.dataset.name

    @property
    def prefix(self):
        return '_'.join((self.model_name, self.dataset_name))

    @classmethod
    def parse_config(cls, config):
        models_and_datasets = parse_models_and_datasets(config)
        return [cls(*args) for args in models_and_datasets]
