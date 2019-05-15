import pickle
import subprocess
from pathlib import Path

from eval.config_parser import parse_dataset
from eval.repo import all_datasets
from eval.utils import Dataset


def unpickle(path):
    return pickle.loads(Path(path).read_bytes())


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
        self.train_set_data = unpickle(dataset.train_set)
        self.train_stats = self.CorpusStats(self.train_set_data)
        self.vocab_data = unpickle(dataset.vocabulary)

    @property
    def vocab_size(self):
        return len(self.vocab_data)


if __name__ == '__main__':
    for ds in parse_dataset(all_datasets):
        ds_stats = DatasetStats(ds)
        print(ds.name)
        print('vocab_size {}'.format(ds_stats.vocab_size))
        print('train.n_examples {}'.format(ds_stats.train_stats.n_examples))
        print('train.n_tokens {}'.format(ds_stats.train_stats.n_tokens))
        print()
