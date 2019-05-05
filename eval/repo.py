import re
from pathlib import Path
from eval.utils import Dataset, Model
from eval.config import models, datasets
from eval.consts import CONTEXTS, REFERENCES, SERBAN_UBUNTU_MODEL_DIR


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
