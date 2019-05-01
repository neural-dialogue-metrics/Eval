from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
import embedding_based as eb
import lsdscc
import rouge
from distinct_n import distinct_n_sentence_level

import numpy as np
import logging
import json

logger = logging.getLogger(__name__)

metrics_classes = {}

CONTEXTS = 'contexts'
RESPONSES = 'responses'
REFERENCES = 'references'
LIST_OF_REFERENCES = 'list_of_references'
EMBEDDINGS = 'embeddings'

ADEM_MODEL = 'adem_model'
RAW_CONTEXTS = 'contexts'
RAW_RESPONSES = 'responses'
RAW_REFERENCES = 'references'


def register_metric(cls):
    def _register():
        metrics_classes[cls.name] = cls
        return cls

    return _register


class Score:

    def __init__(self, name, utterance, system=None, params=None, variant=None):
        self.name = name
        self.utterance = utterance
        self.system = system
        self.params = params
        self.variant = variant

    def json(self, writer):
        json.dump(self.__dict__, writer)


class MetricWrapper:
    requires = None
    init_requires = None
    name = None
    variant = None
    params = None

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def run(self, sess):
        requires = self.requires
        kwargs = {require: sess.load(require) for require in requires}
        score = self(**kwargs)
        sess.save(score)


@register_metric
class BleuScore(MetricWrapper):
    cherry = SmoothingFunction()
    name = 'bleu'
    requires = (RESPONSES, REFERENCES, LIST_OF_REFERENCES)

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
        instances = [cls(n, smoothing) for n in n_list]
        return instances


@register_metric
class EmbeddingBasedScore(MetricWrapper):
    name = 'embedding_based'
    requires = (RESPONSES, REFERENCES, EMBEDDINGS)
    variants = {
        'vector_average': (eb.average_sentence_level, eb.average_corpus_level),
        'vector_extrema': (eb.extrema_sentence_level, eb.extrema_corpus_level),
        'greedy_matching': (eb.greedy_match_sentence_level, eb.greedy_match_corpus_level)
    }

    def __init__(self, variant, sentence_level, corpus_level):
        self.variant = variant
        self.sentence_level = sentence_level
        self.corpus_level = corpus_level

    def __call__(self, responses, references, embeddings):
        utterance = [
            self.sentence_level(hypo, ref, embeddings) for hypo, ref in zip(responses, references)
        ]
        system = self.corpus_level(responses, references, embeddings)
        return utterance, system

    @classmethod
    def new(cls, variant):
        args = cls.variants[variant]
        return cls(variant, *args)

    @classmethod
    def parse_config(cls, config):
        variants = config['variants']


@register_metric
class RougeScore(MetricWrapper):
    name = 'rouge'
    requires = (REFERENCES, RESPONSES)
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


@register_metric
class DistinctScore(MetricWrapper):
    name = 'distinct_n'
    requires = (RESPONSES,)

    def __init__(self, n):
        self.n = n

    def __call__(self, responses):
        utterance = [distinct_n_sentence_level(s, self.n) for s in responses]
        return utterance


class LSDSCCScore(MetricWrapper):
    HYPOTHESIS_SETS = 'hypothesis_sets'
    REFERENCE_SETS = 'reference_sets'

    variants = ('max_bleu', 'mds', 'pds')
    requires = (HYPOTHESIS_SETS, REFERENCE_SETS)

    def __init__(self, variant):
        self.variant = variant

    def __call__(self, hypothesis_sets, reference_sets):
        utterance = [
            lsdscc.compute_score_on_hypothesis_set(h, r) for h, r in zip(hypothesis_sets, reference_sets)
        ]


@register_metric
class ADEMScore(MetricWrapper):
    name = 'adem'
    requires = (RAW_CONTEXTS, RAW_RESPONSES, RAW_REFERENCES, ADEM_MODEL)
    init_requires = (ADEM_MODEL,)

    def __init__(self, adem_model):
        self.adem_model = adem_model

    def __call__(self, contexts, responses, references):
        utterance = self.adem_model.get_scores(
            contexts=contexts,
            gt_responses=references,
            model_responses=responses,
        )
        return utterance


class RuberScore(MetricWrapper):
    name = 'ruber'

    def __init__(self):
        pass
