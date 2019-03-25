"""
Builtin Metric class.
"""
from evaluation.metric.metric_meta import MetricMeta
from evaluation.metric.metric_meta import Signature as sig

# Import all builtin metrics.
import distinct_n
import embedding_based
import rouge
import bleu.metrics as bleu_metrics


class DistinctN(MetricMeta):
    """
    Wrapper for Distinct-N metric.
    """
    signature = (sig.RESPONSE_CORPUS,)

    def __init__(self, n):
        """
        :param n: Distinct N grams to count.
        """
        self.n = n

    def get_name(self):
        return 'Distinct-%d' % self.n

    def __call__(self, **kwargs):
        return distinct_n.distinct_n_corpus_level(
            sentences=kwargs[sig.RESPONSE_CORPUS],
            n=self.n,
        )

    def to_scalar(self, result):
        return result


class EmbeddingBased(MetricMeta):
    """
    Base class for embedding-based metrics.
    """
    signature = (
        sig.RESPONSE_CORPUS,
        sig.REFERENCE_CORPUS,
        sig.EMBEDDINGS,
    )

    def __init__(self, name, metric_fn):
        self._name = name
        self._metric_fn = metric_fn

    def __call__(self, **kwargs):
        return self._metric_fn(
            hypothesis_corpus=kwargs[sig.RESPONSE_CORPUS],
            reference_corpus=kwargs[sig.REFERENCE_CORPUS],
            embeddings=kwargs[sig.EMBEDDINGS],
        )

    def get_name(self):
        return self._name

    def to_scalar(self, result: embedding_based.CorpusLevelScore):
        return result.mean


class AverageScore(EmbeddingBased):
    """
    Wrapper for average score.
    """

    def __init__(self):
        super().__init__(
            name='average',
            metric_fn=embedding_based.average_corpus_level,
        )


class GreedyMatchingScore(EmbeddingBased):
    """
    Wrapper for greedy-matching score.
    """

    def __init__(self):
        super().__init__(
            name='greedy-matching',
            metric_fn=embedding_based.greedy_match_corpus_level,
        )


class ExtremaScore(EmbeddingBased):
    """
    Wrapper for extrema score.
    """

    def __init__(self):
        super().__init__(
            name='extrema',
            metric_fn=embedding_based.extrema_corpus_level,
        )


class Rouge(MetricMeta):
    """
    Base class for ROUGE metrics.
    """
    signature = (
        sig.RESPONSE_CORPUS,
        sig.REFERENCE_CORPUS,
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
        pairs = zip(kwargs[sig.RESPONSE_CORPUS], kwargs[sig.REFERENCE_CORPUS])
        values = [self._apply_metric_fn(s, r) for s, r in pairs]
        return sum(values) / len(values)

    def to_scalar(self, result):
        return result


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


class BleuScore(MetricMeta):
    """
    Wrapper for the BLEU metric.
    """
    signature = (
        sig.REFERENCE_CORPUS,
        sig.RESPONSE_CORPUS,
    )

    def __init__(self, max_order, smooth):
        self.max_order = max_order
        self.smooth = smooth

    def __call__(self, **kwargs):
        return bleu_metrics.compute_bleu(
            translation_corpus=kwargs[sig.RESPONSE_CORPUS],
            reference_corpus=kwargs[sig.REFERENCE_CORPUS],
            max_order=self.max_order,
            smooth=self.smooth,
        )

    def get_name(self):
        return 'BLEU-%d' % self.max_order

    def to_scalar(self, result: bleu_metrics.BleuScore):
        return result.bleu
