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
    'path': lambda s: Path(s),
}

default_load_info = {
    # key, source, format.
    RESPONSES: ('model.responses', 'token_list'),
    CONTEXTS: ('dataset.contexts', 'token_list'),
    REFERENCES: ('dataset.references', 'token_list'),
    EMBEDDINGS: ('metric.embeddings_file', 'embeddings'),
}


def normalize_format(format):
    if callable(format):
        return format
    return loader_fns[format]


class ResourceLoader:

    def __init__(self):
        # (filename, format) => resource
        self.loaded_resources = {}
        # (key, id(requires) => source, format
        self.requires_cache = {}

    # requires can be a dict or a list.
    # if list, the item must be key in default_load_info.
    # if dict, the shape must be similar to default_load_info.
    # the value can be str or callable or list. if str or callable, it is the format.
    # and the source can be found in default_load_info with that key.
    # if list, must be 2 items, the first is the source, the second is the format.
    def get_normalized_requires(self, dict_or_list):
        requires_key = id(dict_or_list)
        if requires_key in self.requires_cache:
            return self.requires_cache[requires_key]

        requires = {}
        if isinstance(dict_or_list, dict):
            for key, value in dict_or_list.items():
                if isinstance(value, str) or callable(value):
                    source, format = default_load_info[key][0], value
                elif isinstance(value, (tuple, list)):
                    source, format = value
                else:
                    raise TypeError('invalid type for requires entry')
                format = normalize_format(format)
                requires[key] = source, format
        elif isinstance(dict_or_list, (list, tuple)):
            for key in dict_or_list:
                source, format = default_load_info[key]
                requires[key] = source, normalize_format(format)
        else:
            raise TypeError('invalid type for requires')

        return self.requires_cache.setdefault(requires_key, requires)

    def load_resources(self, under_test):
        requires = under_test.metric.requires
        requires = self.get_normalized_requires(requires)
        resources = {}
        for key in requires:
            data = self.load_resource_for_key(key, under_test, requires)
            if data is None:
                logger.warning('resource {} unavailable'.format(key))
                return None
            resources[key] = data
        return resources

    def load_resource_for_key(self, key, under_test, requires):
        assert isinstance(requires, dict)
        source, load_fn = requires[key]
        filename = under_test.get_resource_file(source)

        resource_key = (filename, format)
        if resource_key in self.loaded_resources:
            return self.loaded_resources[resource_key]

        try:
            resource = load_fn(filename)
        except Exception:
            traceback.print_exc()
            logging.warning('Exception when loading requires')
            resource = None
        return self.loaded_resources.setdefault(resource_key, resource)

    def get_filenames(self, under_test):
        requires = self.get_normalized_requires(under_test.metric.requires)
        return {
            key: Path(under_test.get_resource_file(source))
            for key, (source, format) in requires.items()
        }

    def get_filename_for_key(self, key, under_test):
        requires = self.get_normalized_requires(under_test.metric.requires)
        source, _ = requires[key]
        return under_test.get_resource_file(source)
