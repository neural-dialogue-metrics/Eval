"""
Builtin Metric class. These are wrappers around the library functions.

Note that each class is a subclass of MetricMeta and declares a signature attribute.
Then in the instance's __call__(), the signature is used to access the parameters in the
kwargs. It is the system that ensures the claimed signature keys are present in the kwargs.
But it is the Metric that ensures the kwargs is used properly i.e., pass the corresponding args
to the underlying functions.
"""

# Import all builtin metrics.
import distinct_n
import embedding_based
import rouge
import bleu

import abc

REFERENCE_CORPUS = 'reference_corpus'
RESPONSE_CORPUS = 'response_corpus'
EMBEDDINGS = 'embeddings'


class MetricValue:
    def __init__(self, name, value):
        """
        >>> MetricValue('BLEU', 0.04)
        <BLEU: 0.0400>

        :param name:
        :param value:
        """
        self.name = name
        self.value = value

    def __str__(self):
        return '%s: %.4f' % (self.name, self.value)

    def __repr__(self):
        return '<Metric %s>' % self


class MetricAdapter(abc.ABC):
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


class DistinctN(MetricAdapter):
    """
    Wrapper for Distinct-N metric.
    """
    signature = (RESPONSE_CORPUS,)

    def __init__(self, n):
        """
        :param n: Distinct N grams to count.
        """
        self.n = n

    def get_name(self):
        return 'Distinct-%d' % self.n

    def __call__(self, **kwargs):
        score = distinct_n.distinct_n_corpus_level(
            sentences=kwargs[RESPONSE_CORPUS],
            n=self.n,
        )
        return MetricValue(self.get_name(), score)


class EmbeddingBased(MetricAdapter):
    """
    Base class for embedding-based metrics.
    """
    signature = (
        RESPONSE_CORPUS,
        REFERENCE_CORPUS,
        EMBEDDINGS,
    )

    def __init__(self, name, metric_fn):
        self._name = name
        self._metric_fn = metric_fn

    def __call__(self, **kwargs):
        score = self._metric_fn(
            hypothesis_corpus=kwargs[RESPONSE_CORPUS],
            reference_corpus=kwargs[REFERENCE_CORPUS],
            embeddings=kwargs[EMBEDDINGS],
        )
        return MetricValue(self._name, score.mean)

    def get_name(self):
        return self._name


class AverageScore(EmbeddingBased):
    """
    Wrapper for average score.
    """

    def __init__(self):
        super().__init__(
            name='Embedding-Average',
            metric_fn=embedding_based.average_corpus_level,
        )


class GreedyMatchingScore(EmbeddingBased):
    """
    Wrapper for greedy-matching score.
    """

    def __init__(self):
        super().__init__(
            name='Greedy-Matching',
            metric_fn=embedding_based.greedy_match_corpus_level,
        )


class ExtremaScore(EmbeddingBased):
    """
    Wrapper for extrema score.
    """

    def __init__(self):
        super().__init__(
            name='Vector-Extrema',
            metric_fn=embedding_based.extrema_corpus_level,
        )


class Rouge(MetricAdapter):
    """
    Base class for ROUGE metrics.
    """
    signature = (
        RESPONSE_CORPUS,
        REFERENCE_CORPUS,
    )

    def _apply_metric_fn(self, summary, reference):
        rouge_score = self._metric_fn(
            summary_sentence=summary,
            reference_sentence=reference,
            alpha=self.alpha,
            **self._kwargs,
        )
        return rouge_score.f1_measure

    def __init__(self, metric_fn, alpha, **kwargs):
        self._metric_fn = metric_fn
        self.alpha = alpha
        self._kwargs = kwargs

    def __call__(self, **kwargs):
        # Run on each pair and then average the result.
        pairs = zip(kwargs[RESPONSE_CORPUS], kwargs[REFERENCE_CORPUS])
        values = [self._apply_metric_fn(s, r) for s, r in pairs]
        score = sum(values) / len(values)
        name = self.get_name()
        return MetricValue(name, score)


class RougeN(Rouge):
    """
    Wrapper for the ROUGE-N.
    """

    def __init__(self, n, alpha=None):
        super().__init__(
            metric_fn=rouge.rouge_n_sentence_level,
            alpha=alpha, n=n,
        )
        self.n = n

    def get_name(self):
        return 'ROUGE-%d' % self.n


class RougeL(Rouge):
    """
    Wrapper for the ROUGE-L.
    """

    def __init__(self, alpha=None):
        super().__init__(
            metric_fn=rouge.rouge_l_sentence_level,
            alpha=alpha,
        )

    def get_name(self):
        return 'ROUGE-L'


class RougeW(Rouge):
    """
    Wrapper for the ROUGE-W.
    """

    def __init__(self, weight=None, alpha=None):
        super().__init__(
            metric_fn=rouge.rouge_w_sentence_level,
            alpha=alpha,
            weight=weight,
        )

    def get_name(self):
        return 'ROUGE-W'


class BleuScore(MetricAdapter):
    """
    Wrapper for the BLEU metric.
    """
    signature = (
        REFERENCE_CORPUS,
        RESPONSE_CORPUS,
    )

    def __init__(self, max_order, smooth):
        self.max_order = max_order
        self.smooth = smooth

    def __call__(self, **kwargs):
        return bleu.compute_bleu(
            translation_corpus=kwargs[RESPONSE_CORPUS],
            reference_corpus=kwargs[REFERENCE_CORPUS],
            max_order=self.max_order,
            smooth=self.smooth,
        )

    def get_name(self):
        return 'BLEU-%d' % self.max_order

    def to_scalar(self, result: bleu.BleuScore):
        return result.bleu
