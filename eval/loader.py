import embedding_based as eb
from pathlib import Path
from eval.consts import *


class ResourceLoader:
    known_loader = {
        # key, source, format.
        RESPONSES: ('model', 'token_list'),
        CONTEXTS: ('dataset', 'token_list'),
        REFERENCES: ('dataset', 'token_list'),
        EMBEDDINGS: ('metric', 'embeddings'),
    }

    def __init__(self):
        self.loaded_resources = {}

    def load_requires(self, under_test):
        requires = under_test.metric.requires
        if isinstance(requires, (list, tuple)):
            # only keys are listed.
            known_loader = self.known_loader
        elif isinstance(requires, dict):
            # detailed loader info of keys are given.
            known_loader = requires
        else:
            raise TypeError('invalid type for requires: {}'.format(type(requires)))
        return {key: self.load_for_key(key, under_test, known_loader) for key in requires}

    def load_for_key(self, key, under_test, requires):
        assert isinstance(requires, dict)

        def get_load_info():
            value = requires[key]
            if isinstance(value, (tuple, list)):
                source, format = value
            elif isinstance(value, str):
                format = value
                source = self.known_loader[key][0]
            else:
                raise TypeError('invalid type: {}'.format(type(value)))
            return source, format

        source, format = get_load_info()
        filename = getattr(under_test, key)
        if filename in self.loaded_resources:
            return self.loaded_resources[filename]
        load_fn = getattr(self, 'load_' + format)
        resource = load_fn(filename)
        return self.loaded_resources.setdefault(filename, resource)

    def load_token_list(self, filename):
        with open(filename) as f:
            return [line.split() for line in f]

    def load_embeddings(self, filename):
        return eb.load_word2vec_binary(filename)

    def load_filename(self, filename):
        return Path(filename).absolute()
