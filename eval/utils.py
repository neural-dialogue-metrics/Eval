from pathlib import Path
from eval.consts import SEPARATOR


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


class Model:
    def __init__(self, name, trained_on, responses, **kwargs):
        self.name = name
        self.trained_on = trained_on
        self.responses = responses
        self.__dict__.update(kwargs)

    def __str__(self):
        return f'Model: {self.name}, {self.trained_on}'


class Dataset:
    def __init__(self, name, contexts, references, **kwargs):
        self.name = name
        self.contexts = contexts
        self.references = references
        self.__dict__.update(kwargs)

    def __repr__(self):
        return '<{} {}>'.format(self.__class__.__qualname__, self.name)


class UnderTest:

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
    def parts(self):
        return self.model_name, self.dataset_name, self.metric_name

    @property
    def prefix(self):
        parts = self.parts
        if any(SEPARATOR in part for part in parts):
            raise ValueError('{!r} is not allowed in names'.format(SEPARATOR))
        return SEPARATOR.join(parts)

    def get_resource_file(self, key):
        return eval('self.{}'.format(key))


def load_template(name):
    filename = Path(__file__).with_name('template').joinpath(name)
    return filename.read_text()


def subdirs(path: Path):
    for file in path.iterdir():
        if file.is_dir():
            yield file
