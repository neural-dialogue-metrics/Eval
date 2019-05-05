import re
from pathlib import Path

from eval.consts import CONTEXTS, REFERENCES, SERBAN_UBUNTU_MODEL_DIR
from eval.utils import Dataset, Model, model_path


def get_model(name, trained_on):
    for model in models:
        if name == model.name and trained_on == model.trained_on:
            return model
    raise ValueError('unknown model: {} on {}'.format(name, trained_on))


def get_dataset(name):
    try:
        params = datasets[name]
    except KeyError as e:
        raise ValueError('unknown dataset: {}'.format(name)) from e
    return Dataset(
        name=name,
        references=params[REFERENCES],
        contexts=params[CONTEXTS],
    )


def find_pretrained_serban_model(prefix, trained_on):
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


def find_serban_ubuntu_models():
    return find_pretrained_serban_model(
        prefix=SERBAN_UBUNTU_MODEL_DIR,
        trained_on='ubuntu',
    )


models = [
    model_path('/home/cgsdfc/Result/HRED-VHRED/Ubuntu/VHRED/output.txt'),
    model_path('/home/cgsdfc/Result/HRED-VHRED/Ubuntu/LSTM/output.txt'),
    model_path('/home/cgsdfc/Result/HRED-VHRED/Ubuntu/HRED/output.txt'),
    model_path('/home/cgsdfc/Result/HRED-VHRED/Opensub/HRED/output.txt'),
    model_path('/home/cgsdfc/Result/HRED-VHRED/Opensub/LSTM/output.txt'),
    model_path('/home/cgsdfc/Result/HRED-VHRED/Opensub/VHRED/output.txt'),
]

datasets = {
    'ubuntu': {
        CONTEXTS: '/home/cgsdfc/UbuntuDialogueCorpus/ResponseContextPairs/raw_testing_contexts.txt',
        REFERENCES: '/home/cgsdfc/UbuntuDialogueCorpus/ResponseContextPairs/raw_testing_responses.txt',
    },
    'opensub': {
        CONTEXTS: '/home/cgsdfc/SerbanOpenSubData/dialogue_length3_6/eval/test.context.txt',
        REFERENCES: '/home/cgsdfc/SerbanOpenSubData/dialogue_length3_6/eval/test.response.txt',
    }
}

metrics = {
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

config = {
    'models': models,
    'datasets': datasets,
    'metrics': metrics
}
