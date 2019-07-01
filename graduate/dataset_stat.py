import pickle

from eval.config_parser import parse_dataset
from eval.repo import all_datasets
from eval.utils import Dataset


def unpickle(path):
    return pickle.load(open(path, 'rb'))


class DatasetStats:
    class CorpusStats:
        def __init__(self, corpus):
            self.corpus = corpus
            pass

        @property
        def n_examples(self):
            return len(self.corpus)

        @property
        def n_tokens(self):
            return sum(len(u) for u in self.corpus)

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.train_set_data = None
        self.test_set_data = None
        self.vocab_data = None

    @property
    def vocab_size(self):
        if self.vocab_data is None:
            self.vocab_data = unpickle(self.dataset.vocabulary)
        return len(self.vocab_data)

    @property
    def train_stats(self):
        if self.train_set_data is None:
            self.train_set_data = unpickle(self.dataset.train_set)
        return self.CorpusStats(self.train_set_data)

    @property
    def test_stats(self):
        if self.test_set_data is None:
            self.test_set_data = unpickle(self.dataset.test_dialogues)
        return self.CorpusStats(self.test_set_data)


def train_stats():
    for ds in parse_dataset(all_datasets):
        ds_stats = DatasetStats(ds)
        print(ds.name)
        print('vocab_size {}'.format(ds_stats.vocab_size))
        print('train.n_examples {}'.format(ds_stats.train_stats.n_examples))
        print('train.n_tokens {}'.format(ds_stats.train_stats.n_tokens))
        print()


def test_stat():
    for ds in parse_dataset(all_datasets):
        ds_stats = DatasetStats(ds)
        print(ds.name)
        print('test.n_examples {}'.format(ds_stats.test_stats.n_examples))
        print('test.n_tokens {}'.format(ds_stats.test_stats.n_tokens))
        print()


if __name__ == '__main__':
    test_stat()
