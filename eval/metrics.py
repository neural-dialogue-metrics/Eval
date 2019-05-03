from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

import embedding_based as eb
import rouge
from distinct_n import distinct_n_sentence_level
from eval.consts import *

metrics_classes = {}


def register_metric(cls):
    metrics_classes[cls.name] = cls
    return cls


class MetricWrapper:
    # __call__ requires
    requires = None
    # __init__ requires
    init_requires = None
    # extract this field for each utterance score
    utterance_field = None
    # extract this field for system score.
    system_field = None
    # name corresponding to a class
    name = None
    # name corresponding to an instance (optional)
    variant = None

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def fullname(self):
        parts = []
        if self.name:
            parts.append(self.name)
        if self.variant:
            parts.append(self.variant)
        name = '_'.join(parts)
        if not name:
            raise ValueError('{} has no valid fullname'.format(self.__class__.__name__))
        return name


@register_metric
class BleuScore(MetricWrapper):
    name = 'bleu'
    cherry = SmoothingFunction()
    requires = (RESPONSES, REFERENCES)

    def __init__(self, n, smoothing):
        self.n = n
        self.smoothing = smoothing
        self._weights = [1 / n for _ in range(n)]
        self._smoothing_fn = self.cherry.method1 if smoothing else self.cherry.method0

    def __call__(self, responses, references):
        list_of_references = [[ref] for ref in references]
        utterance = [sentence_bleu(ref, hypo, self._weights, smoothing_function=self._smoothing_fn)
                     for ref, hypo in zip(list_of_references, responses)]
        system = corpus_bleu(list_of_references, responses, self._weights)
        return utterance, system

    @classmethod
    def parse_config(cls, config):
        # in case it is forgotten, always smoothing.
        smoothing = config.get('smoothing', True)
        n_list = config['n']
        for n in n_list:
            yield cls(n, smoothing)

    @property
    def fullname(self):
        return '_'.join((self.name, str(self.n)))


@register_metric
class EmbeddingBasedScore(MetricWrapper):
    name = 'embedding_based'
    requires = (RESPONSES, REFERENCES, EMBEDDINGS)
    system_field = 'mean'
    variants = {
        'vector_average': (eb.average_sentence_level, eb.average_corpus_level),
        'vector_extrema': (eb.extrema_sentence_level, eb.extrema_corpus_level),
        'greedy_matching': (eb.greedy_match_sentence_level, eb.greedy_match_corpus_level)
    }

    def __init__(self, variant, embeddings_file, sentence_level, corpus_level):
        self.variant = variant
        self.embeddings_file = embeddings_file
        self.sentence_level = sentence_level
        self.corpus_level = corpus_level

    def __call__(self, responses, references, embeddings=None):
        if embeddings is None:
            embeddings = self.load_embeddings(self.embeddings_file)
        utterance = [
            self.sentence_level(hypo, ref, embeddings) for hypo, ref in zip(responses, references)
        ]
        system = self.corpus_level(responses, references, embeddings)
        return utterance, system

    @classmethod
    def new(cls, variant, embeddings_file):
        args = cls.variants[variant]
        return cls(variant, embeddings_file, *args)

    @classmethod
    def parse_config(cls, config):
        for v in config['variants']:
            yield cls.new(v, config['embeddings'])


@register_metric
class RougeScore(MetricWrapper):
    name = 'rouge'
    requires = (REFERENCES, RESPONSES)
    utterance_field = 'f1_measure'
    system_field = utterance_field
    variants = {
        'rouge_n': (rouge.rouge_n_sentence_level, rouge.rouge_n_summary_level),
        'rouge_l': (rouge.rouge_l_sentence_level, rouge.rouge_l_summary_level),
        'rouge_w': (rouge.rouge_w_sentence_level, rouge.rouge_w_summary_level),
    }

    def __init__(self, variant, sentence_level, corpus_level, params):
        self.variant = variant
        self.params = params
        self.sentence_level = lambda s, r: sentence_level(s, r, **params)
        self.corpus_level = lambda s, r: corpus_level(s, r, **params)

    def __call__(self, responses, references):
        utterance = [
            self.sentence_level(sum, ref) for sum, ref in zip(responses, references)
        ]
        system = self.corpus_level(responses, references)
        return utterance, system

    @classmethod
    def new(cls, variant, **kwargs):
        fn_args = cls.variants[variant]
        return cls(variant, *fn_args, params=kwargs)

    @classmethod
    def parse_config(cls, config):
        alpha = config.get('alpha')
        for variant in config['variants']:
            if variant == 'rouge_n':
                for n in config['n']:
                    yield cls.new(variant, alpha=alpha, n=n)
            elif variant == 'rouge_l':
                yield cls.new(variant, alpha=alpha)
            elif variant == 'rouge_w':
                yield cls.new(variant, alpha=alpha, weight=config.get('weight'))

    @property
    def fullname(self):
        if self.variant == 'rouge_n':
            return 'rouge_{}'.format(self.params['n'])
        return self.variant


@register_metric
class DistinctScore(MetricWrapper):
    name = 'distinct_n'
    requires = (RESPONSES,)

    def __init__(self, n):
        self.n = n

    def __call__(self, responses):
        utterance = [distinct_n_sentence_level(s, self.n) for s in responses]
        return utterance, None

    @classmethod
    def parse_config(cls, config):
        for n in config['n']:
            yield cls(n)

    @property
    def fullname(self):
        return 'distinct_{}'.format(self.n)
