import logging
import re
import subprocess
from pathlib import Path

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

import embedding_based as eb
import lsdscc
import rouge
from distinct_n import distinct_n_sentence_level
from eval.consts import *
from eval.utils import load_template

logger = logging.getLogger(__name__)
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

    @classmethod
    def parse_config(cls, config):
        yield cls()

    def compatible(self, model, dataset):
        return True


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
        embeddings = config.get('embeddings', GOOGLE_NEWS_300_BIN)
        for v in config['variants']:
            yield cls.new(v, embeddings)


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
        # this is too slow.
        # system = self.corpus_level(responses, references)
        system = None
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


@register_metric
class UtteranceLenScore(MetricWrapper):
    name = 'utterance_len'
    requires = (RESPONSES,)

    def __call__(self, responses):
        utterance = [len(r) for r in responses]
        return utterance, None


@register_metric
class ADEMScore(MetricWrapper):
    name = 'adem'
    requires = {
        CONTEXTS: 'filename',
        REFERENCES: 'filename',
        RESPONSES: 'filename',
    }

    from eval.consts import ADEM_IMAGE, ADEM_ROOT, ADEM_OUTPUT_FILE
    OUTPUT_FILE = Path(ADEM_ROOT).joinpath(ADEM_OUTPUT_FILE)
    TEMPLATE = load_template(name)

    def __call__(self, contexts, references, responses):
        cmd = self.TEMPLATE.format(
            contexts=contexts,
            references=references,
            responses=responses,
            adem_image=self.ADEM_IMAGE,
            adem_root=self.ADEM_ROOT,
        )
        logger.info('running cmd {}'.format(cmd))
        subprocess.check_call(cmd, shell=True, universal_newlines=True)
        file_content = self.OUTPUT_FILE.read_text()
        return list(map(float, file_content.splitlines())), None


@register_metric
class METEORScore(MetricWrapper):
    name = 'meteor'
    requires = {
        REFERENCES: 'filename',
        RESPONSES: 'filename',
    }

    # Segment 18920 score:    0.33113214948831016
    SEGMENT_SCORE_RE = re.compile(r'^Segment \d+ score:\s+(.*)$', flags=re.MULTILINE)
    # Final score:            0.1634213080811731
    SYSTEM_SCORE_RE = re.compile(r'Final score:\s+(.*)')

    from eval.consts import METEOR_JAR_FILE
    METEOR_JAR_FILE = Path(METEOR_JAR_FILE)
    TEMPLATE = load_template(name)

    def __call__(self, references, responses):
        cmd = self.TEMPLATE.format(
            jar_file=self.METEOR_JAR_FILE,
            references=references,
            responses=responses,
        )
        logger.info('cmd: {}'.format(cmd))
        text = subprocess.check_output(cmd,
                                       shell=True, cwd=self.METEOR_JAR_FILE.parent, universal_newlines=True)
        utterance = self.SEGMENT_SCORE_RE.findall(text)
        utterance = list(map(float, utterance))
        system = self.SYSTEM_SCORE_RE.search(text)
        system = float(system.group(1))
        return utterance, system


@register_metric
class LSDSCCScore(MetricWrapper):
    name = 'lsdscc'
    utterance_field = ('max_bleu', 'pds', 'mds')

    requires = {
        # key, source, load_fn.
        MULTI_RESPONSES: ('model.multi_responses', lsdscc.HypothesisSet.load_corpus),
        REFERENCES: ('metric.references', lsdscc.ReferenceSet.load_json_corpus),
    }

    from lsdscc.align import NLTKBleuAligner, NLTKNistAligner
    aligner_map = {
        'bleu': NLTKBleuAligner,
        'nist': NLTKNistAligner,
    }

    def __init__(self, aligner=None, refset=None):
        aligner_cls = self.aligner_map.get(aligner)
        self.aligner = aligner_cls() if aligner_cls else None
        self.refset = refset

    def __call__(self, multi_responses, references):
        utterance = [
            lsdscc.compute_score_on_hypothesis_set(
                hypothesis_set=h,
                reference_set=r,
                aligner=self.aligner,
            ) for h, r in zip(multi_responses, references)
        ]
        system = lsdscc.compute_score_on_corpus(
            hypothesis_corpus=multi_responses,
            reference_corpus=references,
            aligner=self.aligner,
        )
        return utterance, system

    @property
    def references(self):
        return self.refset or lsdscc.default_reference_set

    @classmethod
    def parse_config(cls, config):
        yield cls(config.get('aligner'), config.get('refset'))

    def compatible(self, model, dataset):
        return hasattr(model, MULTI_RESPONSES) and dataset == 'lsdscc'


@register_metric
class SerbanModelPPLScore(MetricWrapper):
    name = 'serban_ppl'

    SERBAN_MODELS = ('hred', 'vhred', 'lstm')

    # utterance word-perplexity = 71.5183944320154
    UTTER_PPL_RE = re.compile(r'^utterance word-perplexity = (.*)$', flags=re.MULTILINE)
    SYS_PPL_RE = re.compile(r'system word-perplexity = (.*)')

    TEMPLATE = load_template(name)

    requires = {
        TEST_DIALOGUES: ('dataset.test_dialogues', 'filename'),
        MODEL_WEIGHTS: ('model.weights', 'filename'),
    }

    def __init__(self, remove_stopwords=False):
        self.remove_stopwords = remove_stopwords

    def __call__(self, model_weights: Path, test_dialogues: Path):
        cmd = self.TEMPLATE.format(
            model_prefix=model_weights.name.replace('_model.npz', ''),
            save_dir=model_weights.parent,
            test_path=test_dialogues,
            remove_stopwords='-e' if self.remove_stopwords else '',
        )
        logger.info('cmd: {}'.format(cmd))
        text = subprocess.check_output(cmd, shell=True, universal_newlines=True)
        utterance = list(map(float, self.UTTER_PPL_RE.findall(text)))
        system = float(self.SYS_PPL_RE.search(text).group(1))
        return utterance, system

    @classmethod
    def parse_config(cls, config):
        yield cls(
            remove_stopwords=config.get('remove_stopwords')
        )

    @property
    def fullname(self):
        if self.remove_stopwords:
            return self.name + '_no_stopwords'
        return self.name
