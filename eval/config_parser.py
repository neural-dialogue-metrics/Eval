import itertools
#from eval.metrics import metrics_classes
from eval.utils import Model, Dataset


# def parse_metrics(config):
#     metrics = []
#     for name, metric_config in config.items():
#         cls = metrics_classes[name]
#         metrics.extend(cls.parse_config(metric_config))
#     return metrics


def parse_dataset(config):
    dataset = []
    for name, value in config.items():
        dataset.append(Dataset(name, value['context'], value['reference']))
    return dataset


def parse_models(config):
    models = []
    for data_path in config:
        models.append(Model(data_path['name'], data_path['dataset'], data_path['output']))
    return models


# def parse_config(config):
#     metrics = parse_metrics(config['metrics'])
#     models_and_datasets = parse_models_and_datasets(config)
#     return [
#         (metric, model, dataset)
#         for metric, (model, dataset) in itertools.product(metrics, models_and_datasets)
#     ]


def parse_models_and_datasets(config):
    models = parse_models(config['models'])
    datasets = parse_dataset(config['datasets'])

    ds_names = set(ds.name for ds in datasets)
    for model in models:
        if model.trained_on not in ds_names:
            raise ValueError('model {} trained on unknown dataset {}'.format(model.name, model.trained_on))

    return [
        (model, dataset) for model, dataset in itertools.product(models, datasets)
        if model.trained_on == dataset.name
    ]


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
