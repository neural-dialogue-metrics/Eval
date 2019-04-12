import abc


class Signature:
    """
    Standard keys in a signature.
    """
    REFERENCE_CORPUS = 'reference_corpus'
    RESPONSE_CORPUS = 'response_corpus'
    EMBEDDINGS = 'embeddings'


class MetricMeta(abc.ABC):
    def get_name(self):
        """
        Return the canonical name.

        :return: str.
        """
        raise NotImplementedError

    @property
    def signature(self):
        """
        This is a class attribute that tells the system how to invoke the metric.

        :return:
        """
        raise NotImplementedError

    def __call__(self, **kwargs):
        """
        Calling a metric compute the score given kwargs described in signature.

        :param kwargs: A dict with keys in Signature.
        :return:
        """
        raise NotImplementedError

    def __repr__(self):
        return '<Metric %s>' % self.get_name()

    def __str__(self):
        return self.get_name()

    def to_scalar(self, result):
        """
        Turn the result of __call__() into a single float.
        :param result:
        :return: float.
        """
        raise NotImplementedError
