"""
This module handles loading various data from the disk.
"""
from eval.metric_meta import Signature as sig
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
        """
        Load a given resource.

        :param resource: a key in the Signature.
        :param response_path: if resource is RESPONSE_CORPUS, this is the path
        to the corpus.
        :return: the required resource.
        """
        if resource == sig.EMBEDDINGS:
            return self._load_embeddings()
        if resource == sig.REFERENCE_CORPUS:
            return self._load_reference()

        assert resource == sig.RESPONSE_CORPUS
        assert response_path.is_file()
        return self._load_response(response_path)

    def _load_embeddings(self):
        """
        Load the word2vec binary embeddings (mostly for EmbeddingBased).

        :return:
        """
        eb = self._resources.get(sig.EMBEDDINGS, None)
        if eb:
            return eb
        _logger.info('loading embeddings %s', self.embeddings_path)
        eb = load_word2vec_binary(self.embeddings_path)
        return self._resources.setdefault(sig.EMBEDDINGS, eb)

    def _load_reference(self):
        """
        Load the reference corpus.

        :return:
        """
        ref = self._resources.get(sig.REFERENCE_CORPUS, None)
        if ref is not None:
            return ref
        _logger.info('loading reference %s', self.reference_path)
        ref = load_corpus_from_file(self.reference_path)
        return self._resources.setdefault(sig.REFERENCE_CORPUS, ref)

    def _load_response(self, response_path):
        """
        Load the response corpus.

        :return:
        """
        res = self._resources[sig.RESPONSE_CORPUS].get(response_path, None)
        if res is not None:
            return res
        _logger.info('loading response %s', response_path)
        res = load_corpus_from_file(response_path)
        return self._resources[sig.RESPONSE_CORPUS].setdefault(response_path, res)
