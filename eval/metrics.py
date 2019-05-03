from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from distinct_n import distinct_n_sentence_level

import embedding_based as eb
import lsdscc
import rouge

from eval.consts import *

metrics_classes = {}


def register_metric(cls):
    def _register():
        metrics_classes[cls.name] = cls
        return cls

    return _register


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


@register_metric
class BleuScore(MetricWrapper):
    name = 'bleu'
    cherry = SmoothingFunction()
    requires = (RESPONSES, REFERENCES, LIST_OF_REFERENCES)
    utterance_field = 'bleu'
    system_field = 'bleu'

    def __init__(self, n, smoothing):
        self.n = n
        self.smoothing = smoothing
        self._weights = [1 / n for _ in range(n)]
        self._smoothing_fn = self.cherry.method1 if smoothing else self.cherry.method0

    def __call__(self, responses, references, list_of_references=None):
        utterance = [sentence_bleu(ref, hypo, self._weights, smoothing_function=self._smoothing_fn)
                     for ref, hypo in zip(references, responses)]
        if list_of_references is None:
            list_of_references = [[ref] for ref in references]
        system = corpus_bleu(list_of_references, responses, self._weights)
        return utterance, system

    @classmethod
    def parse_config(cls, config):
        # in case it is forgotten, always smoothing.
        smoothing = config.get('smoothing', True)
        n_list = config['n']
        for n in n_list:
            yield cls(n, smoothing)


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

    @classmethod
    def load_embeddings(cls, filename):
        return eb.load_word2vec_binary(filename)


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
        self.sentence_level = lambda s, r: sentence_level(s, r, **params).f1_measure
        self.corpus_level = lambda s, r: corpus_level(s, r, **params).f1_measure

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


@register_metric
class LSDSCCScore(MetricWrapper):
    name = 'lsdscc'
    valid_fields = ('max_bleu', 'mds', 'pds')
    requires = (HYPOTHESIS_SETS, REFERENCE_SETS)

    def __init__(self, fields):
        self.fields = fields

    def __call__(self, hypothesis_sets, reference_sets):
        utterance = [
            lsdscc.compute_score_on_hypothesis_set(h, r)
            for h, r in zip(hypothesis_sets, reference_sets)
        ]
        return utterance, None

    @classmethod
    def parse_config(cls, config):
        for field in config['fields']:
            if field not in cls.valid_fields:
                raise ValueError('invalid field for {}: {}'.format(cls.__name__, field))
        return cls(config['fields'])

    @classmethod
    def load_hypothesis_sets(cls, filename):
        return lsdscc.HypothesisSet.load_corpus(filename)

    @classmethod
    def load_reference_sets(cls):
        return lsdscc.ReferenceSet.load_json_corpus()


@register_metric
class ADEMScore(MetricWrapper):
    name = 'adem'
    requires = (RAW_CONTEXTS, RAW_RESPONSES, RAW_REFERENCES, ADEM_MODEL)
    init_requires = (ADEM_MODEL,)

    def __init__(self, adem_model):
        self.adem_model = adem_model

    def __call__(self, raw_contexts, raw_responses, raw_references):
        utterance = self.adem_model.get_scores(
            contexts=raw_contexts,
            gt_responses=raw_references,
            model_responses=raw_responses,
        )
        return utterance, None


@register_metric
class RuberScore(MetricWrapper):
    name = 'ruber'

    def __init__(self):
        pass
