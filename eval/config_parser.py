import itertools
from eval.metrics import metrics_classes
from eval.utils import Model, Dataset


class ConfigParser:

    def __init__(self):
        pass

    def parse_metrics(self, config):
        metrics = []
        for name, metric_config in config.items():
            cls = metrics_classes[name]
            metrics.extend(cls.parse_config(metric_config))
        return metrics

    def parse_dataset(self, config):
        dataset = []
        for name, value in config.items():
            dataset.append(Dataset(name, value['context'], value['reference']))
        return dataset

    def parse_models(self, config):
        models = []
        for data_path in config:
            models.append(Model(data_path['name'], data_path['dataset'], data_path['output']))
        return models

    def parse_config(self, config):
        metrics = self.parse_metrics(config['metrics'])
        models = self.parse_models(config['models'])
        datasets = self.parse_models(config['datasets'])

        ds_names = set(ds.name for ds in datasets)
        for model in models:
            if model.trained_on not in ds_names:
                raise ValueError('model {} trained on unknown dataset {}'.format(model.name, model.trained_on))

        return [
            (metric, model, dataset)
            for metric, model, dataset in itertools.product(metrics, models, datasets)
            if model.trained_on == dataset.name
        ]
