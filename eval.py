from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
import embedding_based as eb
import lsdscc
import rouge
from distinct_n import distinct_n_sentence_level

import numpy as np
import logging
import json
import itertools

logger = logging.getLogger(__name__)

metrics_classes = {}

CONTEXTS = 'contexts'
RESPONSES = 'responses'
REFERENCES = 'references'
LIST_OF_REFERENCES = 'list_of_references'
EMBEDDINGS = 'embeddings'


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
    fields = None
    params = None

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


@register_metric
class BleuScore(MetricWrapper):
    name = 'bleu'
    cherry = SmoothingFunction()
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
        for n in n_list:
            yield cls(n, smoothing)


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
        for v in variants:
            yield cls.new(v)


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
        return utterance

    @classmethod
    def parse_config(cls, config):
        for n in config['n']:
            yield cls(n)


class LSDSCCScore(MetricWrapper):
    name = 'lsdscc'
    HYPOTHESIS_SETS = 'hypothesis_sets'
    REFERENCE_SETS = 'reference_sets'

    valid_fields = ('max_bleu', 'mds', 'pds')
    requires = (HYPOTHESIS_SETS, REFERENCE_SETS)

    def __init__(self, fields):
        self.fields = fields

    def __call__(self, hypothesis_sets, reference_sets):
        utterance = [
            lsdscc.compute_score_on_hypothesis_set(h, r)
            for h, r in zip(hypothesis_sets, reference_sets)
        ]
        return utterance

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
    def load_reference_sets(cls, filename):
        return lsdscc.ReferenceSet.load_json_corpus()


@register_metric
class ADEMScore(MetricWrapper):
    name = 'adem'
    ADEM_MODEL = 'adem_model'
    RAW_CONTEXTS = 'raw_contexts'
    RAW_RESPONSES = 'raw_responses'
    RAW_REFERENCES = 'raw_references'

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
        return utterance

    @staticmethod
    def load(filename):
        with open(filename) as f:
            return f.readlines()


class RuberScore(MetricWrapper):
    name = 'ruber'

    def __init__(self):
        pass


resource_info = {
    CONTEXTS: {
        'from': 'dataset',
        'loader': '',
        '': ''
    },
    RESPONSES: {
        'from': 'model',
        'loader': '',
    },
    REFERENCES: {
        'from': 'dataset',
        'loader': '',
    }
}


def register_loader(fn):
    pass


class ConfigParser:

    def __init__(self):
        pass

    def parse_metrics(self, config):
        metrics = []
        for name, metric_config in config.items():
            cls = metrics_classes[name]
            metrics.extend(cls.parse_config(metric_config))
        return metrics

    def parse_dataset(self, config):
        dataset = []
        for name, value in config.items():
            dataset.append(Dataset(name, value['context'], value['reference']))
        return dataset

    def parse_models(self, config):
        models = []
        for data_path in config:
            models.append(Model(data_path['name'], data_path['dataset'], data_path['output']))
        return models

    def parse_config(self, config):
        metrics = self.parse_metrics(config['metrics'])
        models = self.parse_models(config['models'])
        datasets = self.parse_models(config['datasets'])

        ds_names = set(ds.name for ds in datasets)
        for model in models:
            if model.trained_on not in ds_names:
                raise ValueError('model {} trained on unknown dataset {}'.format(model.name, model.trained_on))

        return [
            (metric, model, dataset)
            for metric, model, dataset in itertools.product(metrics, models, datasets)
            if model.trained_on == dataset.name
        ]


class Model:
    provides = (RESPONSES,)

    def __init__(self, name, trained_on, responses):
        self.name = name
        self.trained_on = trained_on
        self.responses = responses


class Dataset:
    provides = (CONTEXTS, REFERENCES)

    def __init__(self, name, contexts, references):
        self.name = name
        self.contexts = contexts
        self.references = references

    @classmethod
    def load_contexts(cls, contexts):
        pass


class UnderTest:
    def __init__(self, metric, model, dataset):
        assert model.trained_on == dataset
        self.metric = metric
        self.model = model
        self.dataset = dataset
        self.location = {}

    def resolve(self, key):
        if key in self.location:
            return self.location[key]
        # first see if it is in dataset or model.
        model_or_ds = (self.model, self.dataset)
        for thing in model_or_ds:
            if key in thing.provides:
                file = getattr(thing, key)
                loader = getattr(thing, 'load_' + key)
                return self.location.setdefault(key, (file, loader))
        # try metrics
        if hasattr(self.metric, key):
