import abc


class Signature:
    REFERENCE_CORPUS = 'reference_corpus'
    RESPONSE_CORPUS = 'response_corpus'
    EMBEDDINGS = 'embeddings'


class MetricMeta(abc.ABC):
    def get_name(self):
        raise NotImplementedError

    @property
    def signature(self):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return '<Metric %s>' % self.get_name()

    def __str__(self):
        return self.get_name()

    def to_scalar(self, result):
        raise NotImplementedError
