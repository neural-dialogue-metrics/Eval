"""
Visualization scripts for my submission to ICONIP2019.
"""

import functools
from pathlib import Path
import pickle
from eval.utils import make_parent_dirs
import logging

logger = logging.getLogger(__name__)
SAVE_ROOT = Path('/home/cgsdfc/Metrics/Eval/data/v2/iconip/system')


def cache_this(cache_path, load=pickle.load, dump=pickle.dump):
    """
    A decorator that implements a function with optional cache on the filesystem.

    :param fn: fn does the real computation.
    :param cache_path: used to store the serialized results of fn.
    :param load: `load(path)` -> deserialized object.
    :param dump: `dump(obj, path)` -> serialize object to path.
    :return:
    """
    cache_path = Path(cache_path)

    def decorate(fn):
        @functools.wraps(fn)
        def impl(*args, use_cache=True, **kwargs):
            if cache_path.is_file() and use_cache:
                logger.debug('cache hit, load from {}'.format(cache_path))
                return load(cache_path.open('rb'))
            logger.debug('calling {}'.format(fn.__name__))
            data = fn(*args, **kwargs)
            make_parent_dirs(cache_path)
            dump(data, cache_path.open('wb'))
            return data

        return impl

    return decorate
