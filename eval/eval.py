import argparse
import numpy as np
import logging
import json
import itertools
from consts import *

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

    def load_adem_model(self, *args, **kwargs):
        pass


class UnderTest:
    def __init__(self, metric, model, dataset):
        assert model.trained_on == dataset
        self.metric = metric
        self.model = model
        self.dataset = dataset

    def components(self):
        yield self.model.name
        yield self.dataset
        yield self.metric.name
        if self.metric.variant:
            yield self.metric.variant
        if self.metric.field:
            yield


class Exporter:
    def __init__(self, save_dir):
        self.save_dir = save_dir

    def get_basename(self, under_test):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='path to the config file')
    parser.add_argument('-p', '--prefix', help='path to output files')
    args = parser.parse_args()

    pass
