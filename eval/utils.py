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


def model_path(response_path):
    parts = Path(response_path).parts
    assert parts[-1].endswith('.txt'), 'path not pointing to valid output.txt'
    dataset, model = parts[-3:-1]
    return Model(
        name=model.lower(),
        trained_on=dataset.lower(),
        responses=response_path,
    )


class Model:
    def __init__(self, name, trained_on, responses):
        self.name = name
        self.trained_on = trained_on
        self.responses = responses

    def __repr__(self):
        return '<{} {} on {}>'.format(
            self.__class__.__qualname__, self.name, self.trained_on)


class Dataset:
    def __init__(self, name, contexts, references):
        self.name = name
        self.contexts = contexts
        self.references = references

    def __repr__(self):
        return '<{} {}>'.format(self.__class__.__qualname__, self.name)


class UnderTest:
    SEPARATOR = '-'

    def __init__(self, metric, model, dataset):
        if model.trained_on != dataset.name:
            raise ValueError('model {} was not trained on dataset {}'.format(model.name, dataset.name))
        self.metric = metric
        self.model = model
        self.dataset = dataset

    def __repr__(self):
        return f'<{self.__class__.__qualname__}: {self.model_name}, {self.dataset_name}, {self.metric_name}>'

    @property
    def model_name(self):
        return self.model.name

    @property
    def dataset_name(self):
        return self.dataset.name

    @property
    def metric_name(self):
        return self.metric.fullname

    @property
    def prefix(self):
        return self.SEPARATOR.join((self.model_name, self.dataset_name, self.metric_name))

    @property
    def contexts(self):
        return self.dataset.contexts

    @property
    def references(self):
        return self.dataset.references

    @property
    def responses(self):
        return self.model.responses

    @property
    def embeddings(self):
        return getattr(self.metric, 'embeddings_file')
