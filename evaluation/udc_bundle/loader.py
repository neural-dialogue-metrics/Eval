"""
This module handles loading various data from the disk.
"""
from evaluation.metric.metric_meta import Signature as sig
from evaluation.udc_bundle import EMBEDDING_PATH, REFERENCE_CORPUS_PATH
from embedding_based import load_corpus_from_file
from embedding_based import load_word2vec_binary

import pathlib
import logging

_logger = logging.getLogger(__name__)


class Loader:
    """
    Lazily load component in signature.
    """

    def __init__(self, embeddings_path, reference_path):
        self.embeddings_path = embeddings_path
        self.reference_path = reference_path
        self._resources = {
            sig.RESPONSE_CORPUS: {}
        }

    def load(self, resource, response_path: pathlib.Path = None):
        if resource == sig.EMBEDDINGS:
            return self._load_embeddings()
        if resource == sig.REFERENCE_CORPUS:
            return self._load_reference()

        assert resource == sig.RESPONSE_CORPUS
        assert response_path.is_file()
        return self._load_response(response_path)

    def _load_embeddings(self):
        eb = self._resources.get(sig.EMBEDDINGS, None)
        if eb:
            return eb
        _logger.info('loading embeddings %s', self.embeddings_path)
        eb = load_word2vec_binary(self.embeddings_path)
        return self._resources.setdefault(sig.EMBEDDINGS, eb)

    def _load_reference(self):
        ref = self._resources.get(sig.REFERENCE_CORPUS, None)
        if ref is not None:
            return ref
        _logger.info('loading reference %s', self.reference_path)
        ref = load_corpus_from_file(self.reference_path)
        return self._resources.setdefault(sig.REFERENCE_CORPUS, ref)

    def _load_response(self, response_path):
        res = self._resources[sig.RESPONSE_CORPUS].get(response_path, None)
        if res is not None:
            return res
        _logger.info('loading response %s', response_path)
        res = load_corpus_from_file(response_path)
        return self._resources[sig.RESPONSE_CORPUS].setdefault(response_path, res)

    @classmethod
    def create_for_udc(cls):
        return cls(
            embeddings_path=EMBEDDING_PATH,
            reference_path=REFERENCE_CORPUS_PATH,
        )
