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
