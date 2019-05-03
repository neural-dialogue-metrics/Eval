import argparse
import json
import logging
import numbers
from pathlib import Path

import numpy as np

from eval.consts import *
from eval.config_parser import parse_models_and_datasets

logger = logging.getLogger(__name__)


class ResourceLoader:
    known_loader = {
        RESPONSES: ('model', 'token_list'),
        CONTEXTS: ('dataset', 'token_list'),
        REFERENCES: ('dataset', 'token_list'),
        RAW_CONTEXTS: ('dataset', 'str_list'),
        RAW_REFERENCES: ('dataset', 'str_list'),
        RAW_RESPONSES: ('model', 'str_list'),

        ADEM_MODEL: ('metric', 'adem_model'),
        EMBEDDINGS: ('metric', 'embeddings'),
        HYPOTHESIS_SETS: ('model',),
        REFERENCE_SETS: ('metric',),
    }

    def __init__(self):
        pass

    def load(self, key, under_test):
        source, format = self.known_loader[key]
        if source in ('model', 'dataset'):
            source = getattr(under_test, source)
            filename = getattr(source, key)
            load_fn = getattr(self, 'load_' + format)
            return load_fn(filename)
        # source is metric
        metric = under_test.metric
        load_fn = getattr(metric, 'load_' + key)
        return load_fn()

    def load_token_list(self, filename):
        with open(filename) as f:
            return [line.split() for line in f]

    def load_str_list(self, filename):
        with open(filename) as f:
            return f.readlines()


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


class Exporter:

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

    def export_json(self, under_test, result):
        result = self.process_result(under_test.metric, result)
        prefix = under_test.prefix
        output_path = self.save_dir.joinpath(prefix).with_suffix('.json')
        with output_path.open('w') as f:
            json.dump(result, f, default=self.default)


class Engine:

    def __init__(self, config, save_dir):
        self.exporter = Exporter(save_dir)
        self.under_tests =


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='path to the config file')
    parser.add_argument('-p', '--prefix', help='path to output files')
    args = parser.parse_args()

