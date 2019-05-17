from pylatex import *
from corr.normalize import normalize_name


class TableMaker:

    def __init__(self):
        pass


if __name__ == '__main__':
    print(normalize_name('metric', 'bleu-4'))
    print(normalize_name('metric', 'bleu-4'))
    print(normalize_name('metric', 'bleu-4'))
    print(normalize_name('metric', 'bleu-4'))
    print(normalize_name('metric', 'bleu-4'))
    print(normalize_name('model', 'hred'))
    print(normalize_name('dataset', 'opensub'))
    print(normalize_name('dataset', 'lsdscc'))
    print(normalize_name.cache_info())
