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


default_load_fns = {
    # format name, load_fn.
    'token_list': eb.load_corpus_from_file,
    'embeddings': eb.load_word2vec_binary,
    'filename': load_filename,
    'path': lambda s: Path(s),
    'lines': lambda s: Path(s).read_text().splitlines(),
}

default_requires = {
    # key, source, format.
    RESPONSES: ('model.responses', 'token_list'),
    CONTEXTS: ('dataset.contexts', 'token_list'),
    REFERENCES: ('dataset.references', 'token_list'),
    EMBEDDINGS: ('metric.embeddings_file', 'embeddings'),
}


def normalize_format(format):
    if callable(format):
        return format
    return default_load_fns[format]


class ResourceLoader:

    def __init__(self):
        # (filename, format) => resource
        self.resources_cache = {}
        # id(requires) => normalized_requires
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

        def do_normalize():
            requires = {}
            if isinstance(dict_or_list, dict):
                for key, value in dict_or_list.items():
                    if isinstance(value, str) or callable(value):
                        source, load_fn = default_requires[key][0], value
                    elif isinstance(value, (tuple, list)):
                        source, load_fn = value
                    else:
                        raise TypeError('invalid type for requires entry')
                    load_fn = normalize_format(load_fn)
                    requires[key] = source, load_fn
            elif isinstance(dict_or_list, (list, tuple)):
                for key in dict_or_list:
                    source, load_fn = default_requires[key]
                    requires[key] = source, normalize_format(load_fn)
            else:
                raise TypeError('invalid type for requires')
            return requires

        requires = do_normalize()
        return self.requires_cache.setdefault(requires_key, requires)

    def load_resources(self, under_test):
        requires = self.get_normalized_requires(under_test.metric.requires)
        resources = {}
        for key in requires:
            data = self.load_resource_for_key(key, under_test, requires)
            if data is None:
                logger.warning('resource {} unavailable'.format(key))
                return None
            resources[key] = data
        return resources

    def load_resource_for_key(self, key, under_test, requires):
        source, load_fn = requires[key]
        filename = under_test.get_resource_file(source)
        logger.info('{} resolved to {}'.format(source, filename))
        resource_key = (filename, load_fn)
        if resource_key in self.resources_cache:
            return self.resources_cache[resource_key]

        try:
            resource = load_fn(filename)
        except Exception:
            traceback.print_exc()
            logging.warning('Exception when loading requires')
            resource = None
        return self.resources_cache.setdefault(resource_key, resource)

    def get_filenames(self, under_test):
        requires = self.get_normalized_requires(under_test.metric.requires)
        return {
            key: Path(under_test.get_resource_file(source))
            for key, (source, _) in requires.items()
        }

    def get_filename_for_key(self, key, under_test):
        requires = self.get_normalized_requires(under_test.metric.requires)
        source, _ = requires[key]
        return under_test.get_resource_file(source)
