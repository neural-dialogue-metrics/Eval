import logging
import re
from pathlib import Path

from eval.consts import *
from eval.utils import Dataset, Model, subdirs, SerbanModel

logger = logging.getLogger(__name__)


def get_model(name, trained_on):
    for model in all_models:
        if name == model.name and trained_on == model.trained_on:
            return model
    raise ValueError('unknown model: {} trained on {}'.format(name, trained_on))


def get_dataset(name):
    try:
        params = all_datasets[name]
    except KeyError as e:
        raise ValueError('unknown dataset: {}'.format(name)) from e
    return Dataset(name=name, **params)


def get_config(models=None, datasets=None, metrics=None):
    return {
        'models': models or all_models,
        'datasets': datasets or all_datasets,
        'metrics': metrics or all_metrics,
    }


def find_pretrained_serban_model(prefix=SERBAN_UBUNTU_MODEL_DIR, trained_on='ubuntu'):
    first_txt_re = re.compile(r'_First\.txt$')

    def iter_models():
        for dir in Path(prefix).iterdir():
            model = Model(name=dir.name, trained_on=trained_on, responses=None)
            files = list(dir.glob('*.txt'))
            for file in files:
                if first_txt_re.search(file.name):
                    model.responses = file
                else:
                    model.multi_responses = file
            if not model.responses:
                raise ValueError('no valid responses found for model {}'.format(model.name))
            yield model

    return list(iter_models())


def model_path(response_path):
    parts = Path(response_path).parts
    assert parts[-1].endswith('.txt'), 'path not pointing to valid output.txt'
    dataset, model = parts[-3:-1]
    return Model(
        name=model.lower(),
        trained_on=dataset.lower(),
        responses=response_path,
    )


class SerbanModelFinder:
    SUFFIXES = ('_model.npz', '_timing.npz', '_state.pkl')

    def __init__(self, model_root):
        self.model_root = Path(model_root)

    def find_latest_model(self, model_dir):
        model_suffix = self.SUFFIXES[0]
        model_files = list(model_dir.glob('*' + model_suffix))
        if not model_files:
            raise ValueError('no model file found in {}'.format(model_dir))
        model_files = list(filter(lambda file: '_auto_' not in file.name, model_files))
        model_files = sorted(model_files, key=lambda file: file.stat().st_mtime, reverse=True)
        latest_model: Path = model_files[0]
        logger.info('Found latest npz file: {}'.format(latest_model))

        model_id = latest_model.name.replace(model_suffix, '')
        model_prefix = latest_model.with_name(model_id)
        for suffix in self.SUFFIXES:
            file = model_prefix.with_name(model_id + suffix)
            if not file.exists():
                raise ValueError('file {} does not exist'.format(file))
        return latest_model

    def find_models(self):
        for dataset_dir in subdirs(self.model_root):
            for model_dir in subdirs(dataset_dir):
                weights = self.find_latest_model(model_dir)
                output_file = self.model_root.joinpath(
                    dataset_dir.name).joinpath(model_dir.name).joinpath(OUTPUT_FILENAME)
                if not output_file.exists():
                    logger.warning('output_file {} does not exist'.format(output_file))
                yield SerbanModel(
                    name=model_dir.name.lower(),
                    trained_on=dataset_dir.name.lower(),
                    responses=output_file,
                    weights=weights,
                )


def find_serban_models(model_root=SERBAN_MODEL_ROOT):
    return list(SerbanModelFinder(model_root).find_models())


def find_random_models(model_root=RANDOM_MODEL_ROOT):
    model_root = Path(model_root)
    model_name = model_root.name.lower()

    def iter_models():
        for ds in subdirs(model_root):
            responses = ds.joinpath(OUTPUT_FILENAME)
            yield Model(name=model_name, trained_on=ds.name.lower(), responses=responses)

    return list(iter_models())


all_models = find_serban_models() + find_random_models()

all_datasets = {
    'ubuntu': {
        CONTEXTS: '/home/cgsdfc/UbuntuDialogueCorpus/ResponseContextPairs/raw_testing_contexts.txt',
        REFERENCES: '/home/cgsdfc/UbuntuDialogueCorpus/ResponseContextPairs/raw_testing_responses.txt',
        TEST_DIALOGUES: '/home/cgsdfc/UbuntuDialogueCorpus/Test.dialogues.pkl',
        VOCABULARY: '/home/cgsdfc/UbuntuDialogueCorpus/Dataset.dict.pkl',
        TRAIN_SET: '/home/cgsdfc/UbuntuDialogueCorpus/Training.dialogues.pkl',
    },
    'opensub': {
        CONTEXTS: '/home/cgsdfc/SerbanOpenSubData/dialogue_length3_6/eval/test.context.txt',
        REFERENCES: '/home/cgsdfc/SerbanOpenSubData/dialogue_length3_6/eval/test.response.txt',
        TEST_DIALOGUES: '/home/cgsdfc/SerbanOpenSubData/dialogue_length3_6/eval/test.dialogues.pkl',
        VOCABULARY: '/home/cgsdfc/SerbanOpenSubData/dialogue_length3_6/train.dict.pkl',
        TRAIN_SET: '/home/cgsdfc/SerbanOpenSubData/dialogue_length3_6/train.dialogues.pkl',
    },
    'lsdscc': {
        CONTEXTS: '/home/cgsdfc/SerbanLSDSCC/raw_test_dialogues.context',
        REFERENCES: '/home/cgsdfc/SerbanLSDSCC/raw_test_dialogues.response',
        TEST_DIALOGUES: '/home/cgsdfc/SerbanLSDSCC/vocab_35000/Test.dialogues.pkl',
        VOCABULARY: '/home/cgsdfc/SerbanLSDSCC/vocab_35000/Train.dict.pkl',
        TRAIN_SET: '/home/cgsdfc/SerbanLSDSCC/vocab_35000/Train.dialogues.pkl',
    }
}

all_metrics = {
    'bleu': {
        'n': [1, 2, 3, 4],
        'smoothing': True,
    },
    'rouge': {
        'alpha': 0.9,
        'weight': 1.2,
        'n': [1, 2, 3, 4],
        'variants': [
            'rouge_n',
            'rouge_l',
            'rouge_w',
        ],
    },
    'distinct_n': {
        'n': [1, 2]
    },
    'embedding_based': {
        'variants': [
            'vector_average',
            'vector_extrema',
            'greedy_matching',
        ],
    },
    'adem': {},
    'meteor': {},
    # 'serban_ppl': {},
    'utterance_len': {},
}

default_config = {
    'models': all_models,
    'datasets': all_datasets,
    'metrics': all_metrics
}
