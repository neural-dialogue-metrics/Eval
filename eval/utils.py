from pathlib import Path
import os


def ruber_data(train_dir, data_dir, embedding):
    data_dir = Path(data_dir)
    query_vocab = data_dir.glob('*_contexts.*.vocab*')
    query_embed = data_dir.glob('*_contexts.*.embed')
    reply_vocab = data_dir.glob('*_responses.*.vocab*')
    reply_embed = data_dir.glob('*_responses.*.embed')
    return {
        'train_dir': train_dir,
        'query_vocab': query_vocab,
        'query_embed': query_embed,
        'reply_vocab': reply_vocab,
        'reply_embed': reply_embed,
        'embedding': embedding,
    }


def data_path(response):
    parts = os.path.split(response)
    assert parts[-1].endswith('.txt'), 'path not pointing to valid output.txt'
    dataset, model = parts[-3:-1]
    return {
        'dataset': dataset.lower(),
        'model': model.lower(),
        'output': response,
    }


class Model:
    def __init__(self, name, trained_on, responses):
        self.name = name
        self.trained_on = trained_on
        self.responses = responses


class Dataset:
    def __init__(self, name, contexts, references):
        self.name = name
        self.contexts = contexts
        self.references = references
