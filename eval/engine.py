import argparse
import itertools
import json
import logging
import numbers
import numpy as np
import embedding_based as eb

from pathlib import Path
from eval.consts import *
from eval.config_parser import parse_models_and_datasets, parse_metrics

logger = logging.getLogger(__name__)


class ResourceLoader:
    known_loader = {
        RESPONSES: ('model', 'token_list'),
        CONTEXTS: ('dataset', 'token_list'),
        REFERENCES: ('dataset', 'token_list'),
        EMBEDDINGS: ('metric', 'embeddings'),
    }

    def __init__(self):
        self.loaded_resources = {
            'model': {},
            'dataset': {},
            'metric': {},
        }

    def load(self, key, under_test):
        source, format = self.known_loader[key]
        namespace = getattr(self.loaded_resources, source)
        if key in namespace:
            return namespace[key]

        filename = getattr(under_test, key)
        load_fn = getattr(self, 'load_' + format)
        resource = load_fn(filename)
        return namespace.setdefault(key, resource)

    def load_token_list(self, filename):
        with open(filename) as f:
            return [line.split() for line in f]

    def load_str_list(self, filename):
        with open(filename) as f:
            return f.readlines()

    def load_embeddings(self, filename):
        return eb.load_word2vec_binary(filename)


class UnderTest:
    SEPARATOR = '-'

    def __init__(self, metric, model, dataset):
        assert model.trained_on == dataset
        self.metric = metric
        self.model = model
        self.dataset = dataset

    @property
    def model_name(self):
        return self.model.name

    @property
    def dataset_name(self):
        return self.dataset.name

    @property
    def metric_name(self):
        base = self.metric.name
        if self.metric.variant:
            return self.SEPARATOR.join((base, self.metric.variant))
        return base

    @property
    def prefix(self):
        return self.SEPARATOR.join((self.model_name, self.dataset_name, self.metric_name))

    @property
    def contexts(self):
        return self.dataset.contexts

    @property
    def references(self):
        return self.dataset.references

    @property
    def responses(self):
        return self.model.responses

    @property
    def embeddings(self):
        return getattr(self.metric, 'embeddings_file')


class Exporter:
    CONFIG_JSON = 'config.json'

    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)

    def process_result(self, under_test, result):
        metric = under_test.metric
        utterance, system = result

        def extract_fields(score, fields):
            if not fields:
                if isinstance(score, numbers.Number):
                    return score
                return score.__dict__
            if isinstance(fields, str):
                return getattr(score, fields)
            fields = tuple(fields)
            return {name: getattr(score, name) for name in fields}

        utterance = [extract_fields(score, metric.utterance_field) for score in utterance]
        if system is None:
            system = np.mean(utterance)
        else:
            system = extract_fields(system, metric.system_field)
        return dict(
            utterance=utterance,
            system=system,
            metric=under_test.metric_name,
            model=under_test.model_name,
            dataset=under_test.dataset_name,
        )

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool):
            return bool(obj)
        else:
            raise TypeError

    def export_json(self, result, under_test):
        result = self.process_result(under_test.metric, result)
        prefix = under_test.prefix
        output_path = self.save_dir.joinpath(prefix).with_suffix('.json')
        with output_path.open('w') as f:
            json.dump(result, f, default=self.default)

    def export_config(self, config):
        config_json = self.save_dir.joinpath(self.CONFIG_JSON)
        config_json.write_text(json.dumps(config))


class Engine:
    def __init__(self, config, save_dir):
        self.exporter = Exporter(save_dir)
        self.loader = ResourceLoader()
        self.config = config
        try:
            from eval.metrics import metrics_classes
            self.metrics_classes = metrics_classes
        except ImportError:
            logger.error('metric_classes not available. Some of the packages were not installed?')
            raise
        self.under_tests = self.parse_config(config)

    def parse_config(self, config):
        metrics = parse_metrics(config['metrics'], self.metrics_classes)
        models_and_datasets = parse_models_and_datasets(config)
        return [
            UnderTest(metric=metric, model=model, dataset=dataset)
            for metric, (model, dataset) in itertools.product(metrics, models_and_datasets)
        ]

    def run(self):
        self.exporter.export_config(self.config)
        for under_test in self.under_tests:
            payload = {key: self.loader.load(key, under_test) for key in under_test.metric.requires}
            result = under_test.metric(**payload)
            self.exporter.export_json(result, under_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', help='path to output files')
    args = parser.parse_args()

    from eval.config import config

    engine = Engine(config, args.prefix)
    engine.run()
