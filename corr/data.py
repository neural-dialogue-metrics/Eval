from pathlib import Path

import pandas as pd
from corr.utils import load_filename_data
from corr.utils import UtterScoreDist


class DataIndex:

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir).absolute()
        self._index = None
        self._cache = {}

    @property
    def index(self):
        if not self._index:
            self._index = load_filename_data(self.data_dir)
        return self._index

    def iter_triples(self):
        return self.index.itertuples(index=False, name='Triple')

    def get_data(self, path):
        if path in self._cache:
            return self._cache[path]
        return self._cache.setdefault(path, UtterScoreDist.from_json_file(path))


if __name__ == '__main__':
    data_index = DataIndex('./save')
    for triple in data_index.iter_triples():
        print(triple)
