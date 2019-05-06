import logging
import re
from pathlib import Path

from eval.consts import *
from eval.utils import Dataset, Model, subdirs

logger = logging.getLogger(__name__)


def get_model(name, trained_on):
    for model in all_models:
        if name == model.name and trained_on == model.trained_on:
            return model
    raise ValueError('unknown model: {} on {}'.format(name, trained_on))


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


def model_path(response_path, **kwargs):
    parts = Path(response_path).parts
    assert parts[-1].endswith('.txt'), 'path not pointing to valid output.txt'
    dataset, model = parts[-3:-1]
    return Model(
        name=model.lower(),
        trained_on=dataset.lower(),
        responses=response_path,
        **kwargs,
    )


class SerbanModelFinder:
    SUFFIXES = ('_model.npz', '_timing.npz', '_state.pkl')
    OUTPUT = 'output.txt'

    def __init__(self, model_root, result_root):
        self.model_root = Path(model_root)
        self.result_root = Path(result_root)

    def find_latest_model(self, model_dir):
        model_suffix = self.SUFFIXES[0]
        npz_files = list(model_dir.glob('*' + model_suffix))
        if not npz_files:
            raise ValueError('no model file found in {}'.format(model_dir))
        npz_files = list(filter(lambda file: '_auto_' not in file.name, npz_files))
        npz_files = sorted(npz_files, key=lambda file: file.stat().st_mtime, reverse=True)
        latest_npz: Path = npz_files[0]
        logger.info('Found latest npz file: {}'.format(latest_npz))

        model_id = latest_npz.name.replace(model_suffix, '')
        model_prefix = latest_npz.with_name(model_id)
        for suffix in self.SUFFIXES:
            file = model_prefix.with_name(model_id + suffix)
            if not file.exists():
                raise ValueError('file {} does not exist'.format(file))
        return model_prefix

    def find_models(self):
        for dataset_dir in subdirs(self.model_root):
            for model_dir in subdirs(dataset_dir):
                model_prefix = self.find_latest_model(model_dir)
                output_file = self.result_root.joinpath(
                    dataset_dir.name).joinpath(model_dir.name).joinpath(self.OUTPUT)
                if not output_file.exists():
                    logger.warning('output_file {} does not exist'.format(output_file))
                yield Model(
                    name=model_dir.name.lower(),
                    trained_on=dataset_dir.name.lower(),
                    responses=output_file,
                    weights=model_prefix,
                )


def find_serban_models(model_root=SERBAN_MODEL_ROOT, result_root=SERBAN_RESULT_ROOT):
    return list(SerbanModelFinder(model_root, result_root).find_models())


all_models = [
    model_path('/home/cgsdfc/Result/HRED-VHRED/Ubuntu/VHRED/output.txt',
               weights='/home/cgsdfc/SavedModels/HRED-VHRED/Ubuntu/VHRED/UbuntuModel'),

    model_path('/home/cgsdfc/Result/HRED-VHRED/Ubuntu/LSTM/output.txt',
               weights='/home/cgsdfc/SavedModels/HRED-VHRED/Ubuntu/LSTM/1556265276.0421634_UbuntuModel'),

    model_path('/home/cgsdfc/Result/HRED-VHRED/Ubuntu/HRED/output.txt',
               weights='/home/cgsdfc/SavedModels/HRED-VHRED/Ubuntu/HRED/1554192029.6059802_UbuntuModel'),

    model_path('/home/cgsdfc/Result/HRED-VHRED/Opensub/HRED/output.txt',
               weights=''),

    model_path('/home/cgsdfc/Result/HRED-VHRED/Opensub/LSTM/output.txt'),
    model_path('/home/cgsdfc/Result/HRED-VHRED/Opensub/VHRED/output.txt'),
]

all_datasets = {
    'ubuntu': {
        CONTEXTS: '/home/cgsdfc/UbuntuDialogueCorpus/ResponseContextPairs/raw_testing_contexts.txt',
        REFERENCES: '/home/cgsdfc/UbuntuDialogueCorpus/ResponseContextPairs/raw_testing_responses.txt',
        TEST_DIALOGUES: '/home/cgsdfc/UbuntuDialogueCorpus/Test.dialogues.pkl',
    },
    'opensub': {
        CONTEXTS: '/home/cgsdfc/SerbanOpenSubData/dialogue_length3_6/eval/test.context.txt',
        REFERENCES: '/home/cgsdfc/SerbanOpenSubData/dialogue_length3_6/eval/test.response.txt',
        TEST_DIALOGUES: '/home/cgsdfc/SerbanOpenSubData/dialogue_length3_6/eval/test.dialogues.pkl',
    }
}

all_metrics = {
    'bleu': {
        'n': [4],
        'smoothing': True,
    },
    'rouge': {
        'alpha': 0.9,
        'weight': 1.2,
        'n': [2],
        'variants': ['rouge_n'],
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
}

default_config = {
    'models': all_models,
    'datasets': all_datasets,
    'metrics': all_metrics
}
