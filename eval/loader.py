import traceback

import embedding_based as eb
import logging

from pathlib import Path
from eval.consts import *

logger = logging.getLogger(__name__)


def load_filename(filename):
    filename = Path(filename).absolute()
    if not filename.exists():
        logger.warning('{} not found'.format(filename))
        raise FileNotFoundError
    return filename


loader_fns = {
    'token_list': eb.load_corpus_from_file,
    'embeddings': eb.load_word2vec_binary,
    'filename': load_filename,
}

default_load_info = {
    # key, source, format.
    RESPONSES: ('model.responses', 'token_list'),
    CONTEXTS: ('dataset.contexts', 'token_list'),
    REFERENCES: ('dataset.references', 'token_list'),
    EMBEDDINGS: ('metric.embeddings_file', 'embeddings'),
}


class ResourceLoader:

    def __init__(self):
        self.loaded_resources = {}
        self.requires_cache = {}

    def load_requires(self, under_test):
        requires = under_test.metric.requires
        if isinstance(requires, (list, tuple)):
            # only keys are listed.
            known_loader = default_load_info
        elif isinstance(requires, dict):
            # detailed loader info of keys are given.
            known_loader = requires
        else:
            raise TypeError('invalid type for requires: {}'.format(type(requires)))

        try:
            return {key: self.load_for_key(key, under_test, known_loader) for key in requires}
        except Exception:
            traceback.print_exc()
            logging.warning('Exception when loading requires')
            return None

    def get_load_info(self, key, requires):
        # find out how to load a key in a requires.
        cache_key = (key, id(requires))
        if cache_key in self.requires_cache:
            return self.requires_cache[cache_key]

        value = requires[key]
        if isinstance(value, (tuple, list)):
            source, format = value
        elif isinstance(value, str):
            format = value
            source = default_load_info[key][0]
        else:
            raise TypeError('invalid type: {}'.format(type(value)))

        if callable(format):
            load_fn = format
        elif isinstance(format, str):
            load_fn = loader_fns[format]
        else:
            raise TypeError('invalid type for format: {}'.format(type(format)))
        return self.requires_cache.setdefault(cache_key, (source, load_fn))

    def load_for_key(self, key, under_test, requires):
        assert isinstance(requires, dict)
        source, load_fn = self.get_load_info(key, requires)
        logger.info('source: {}'.format(source))
        logger.info('load_fn: {}'.format(load_fn))

        filename = under_test.get_resource_file(source)
        if filename in self.loaded_resources:
            return self.loaded_resources[filename]

        resource = load_fn(filename)
        return self.loaded_resources.setdefault(filename, resource)
