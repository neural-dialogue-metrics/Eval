import embedding_based as eb
from eval.consts import *


class ResourceLoader:
    known_loader = {
        RESPONSES: ('model', 'token_list'),
        CONTEXTS: ('dataset', 'token_list'),
        REFERENCES: ('dataset', 'token_list'),
        EMBEDDINGS: ('metric', 'embeddings'),
    }

    def __init__(self):
        self.loaded_resources = {}

    def load(self, key, under_test):
        source, format = self.known_loader[key]
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
