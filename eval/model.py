"""
This module handles the discovery of models and their generated responses.
"""
import abc
import logging
import pathlib
from pathlib import Path
import io

from typing import Optional

_logger = logging.getLogger(__name__)


class ModelOutput:
    """
    A bundle of filenames of the output of a model.
    """

    def __init__(self, response_path: Path, nbest_path: Optional[Path] = None):
        self.response_path = response_path
        self.nbest_path = nbest_path

    def __str__(self):
        with io.StringIO() as f:
            print(self.__class__.__name__, file=f)
            print('response:', self.response_path, file=f)
            print('nbest:', self.nbest_path, file=f)
            return f.getvalue()


class ModelInfo:
    """
    A piece of meta information about a model.
    """

    def __init__(self, name, dataset, output):
        self.name = name
        self.dataset = dataset
        self.output = output

    def __str__(self):
        return '%s-%s' % (self.name, self.dataset)


class ModelFinder(abc.ABC):
    def find_models(self):
        """
        Return an iterable of ModelOutput.

        :return:
        """
        raise NotImplementedError


class _DirModelFinder(ModelFinder):
    """
    Find all models under a specific dir.
    The dir is assumed to have a predefined structure.

    Root/
        Model-1/
            first_response
            nbest_list
        Model-2/
            ...

    """

    def __init__(self, root, dataset, is_response=None, is_nbest=None, known_models=None):
        """

        :param root: Root dir.
        :param dataset: name of the dataset.
        :param is_response: a fn determine a filename denotes a response file.
        :param is_nbest: a fn determine a filename denotes a NBest list file.
        :param known_models: a list of model name *not* to skip.
        """
        self._root = root if isinstance(root, pathlib.Path) else pathlib.Path(root)
        self._known_models = known_models
        self._is_response = is_response
        self._is_nbest = is_nbest
        self._dataset = dataset

    def _is_known_model(self, name):
        if self._known_models is None:
            return True
        return name in self._known_models

    def _get_dirs(self):
        return [d for d in self._root.iterdir() if d.is_dir() and self._is_known_model(d.name)]

    def _make_model(self, _dir):
        model_name = _dir.name

        def find_file(files, fn):
            for f in files:
                if fn and fn(f.name):
                    return f
            else:
                return files[0]

        text_files = list(_dir.glob('*.txt'))
        if not text_files:
            raise ValueError('model dir is empty: %s' % _dir)
        output = ModelOutput(
            response_path=find_file(text_files, self._is_response),
            nbest_path=find_file(text_files, self._is_nbest)
        )
        return ModelInfo(model_name, self._dataset, output)

    def find_models(self):
        if not self._root.is_dir():
            raise ValueError('root is not a dir: %s' % self._root)
        dirs = self._get_dirs()
        if not dirs:
            raise ValueError('no model found in %s' % self._root)
        for _dir in dirs:
            yield self._make_model(_dir)


class UbuntuSerbanModelFinder(_DirModelFinder):
    """
    The structure of the Serban's model outputs of Ubuntu is:

    UbuntuDialogueCorpus/
        ResponseContextPairs/
            ModelPredictions/
                Model-1/
                    File.txt -- NBest list

                    File.txt_First.txt -- Response

    """
    FIRST_RESPONSE_SUFFIX = '_First.txt'
    DATASET = 'Ubuntu'
    RESPONSE_CONTEXT_PAIRS = 'ResponseContextPairs'
    MODEL_PREDICTIONS = 'ModelPredictions'
    KNOWN_MODELS = (
        'HRED_Baseline',
        'LSTM_Baseline',
        'VHRED',
    )

    def __init__(self, root):
        root = pathlib.Path(root)
        super().__init__(
            root / self.RESPONSE_CONTEXT_PAIRS / self.MODEL_PREDICTIONS,
            self.DATASET,
            known_models=self.KNOWN_MODELS,
            is_response=lambda s: s.endswith(self.FIRST_RESPONSE_SUFFIX),
            is_nbest=lambda s: not s.endswith(self.FIRST_RESPONSE_SUFFIX)
        )


class TwitterSerbanModelFinder(ModelFinder):
    """
    The structure of the model outputs on Twitter is:

    TwitterDialogueCorpus/
        ModelResponses/
            First_Model_GeneratedTestResponses_First10000.txt

    """
    MODEL_RESPONSES = 'ModelResponses'
    GENERATED_TEST_RESPONSES = 'GeneratedTestResponses'
    DATASET = 'Twitter'

    def __init__(self, root):
        self._root = pathlib.Path(root) / self.MODEL_RESPONSES
        pass

    def find_models(self):
        def get_model_name(path: pathlib.Path):
            name = path.name
            things = name.split('_')
            end = things.index(self.GENERATED_TEST_RESPONSES)
            # things = things[1:end]
            # First_HRED_BeamSearch
            # return '-'.join(things)
            return things[1]

        for file in self._root.glob('*.txt'):
            model_name = get_model_name(file)
            output = ModelOutput(file)
            yield ModelInfo(
                name=model_name,
                dataset=self.DATASET,
                output=output,
            )
