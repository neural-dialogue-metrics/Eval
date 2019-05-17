import re
import subprocess
import logging
from pathlib import Path

from eval.consts import SERBAN_UBUNTU_MODEL_DIR, OUTPUT_FILENAME, SERBAN_MODEL_ROOT, RANDOM_MODEL_ROOT
from eval.utils import subdirs, should_make, load_template, get_random_gpu

logger = logging.getLogger(__name__)


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


class Model:
    def __init__(self, name, trained_on, responses):
        self.name = name
        self.trained_on = trained_on
        self.responses = responses

    def __repr__(self):
        return f'Model({self.name}, {self.trained_on})'


class SerbanModel(Model):

    @classmethod
    def get_prototype(cls, model, dataset):
        dataset = cls.dataset_out_rules.get(dataset, dataset.lower())
        return f'prototype_{dataset}_{model.upper()}'

    dataset_out_rules = {
        'opensub': 'opensubtitles',
    }

    def __init__(self, weights=None, prototype=None, **kwargs):
        super().__init__(**kwargs)
        self.weights = weights
        self._prototype = prototype

    @property
    def prototype(self):
        if self._prototype is None:
            return self.get_prototype(self.name, self.trained_on)
        return self._prototype

    def _get_model_prefix(self):
        weights = self.weights
        return weights.with_name(weights.name.replace('_model.npz', ''))

    def _get_docker_name(self, job):
        return f'{self.name}_{self.trained_on}_{job}'

    def sample(self):
        from eval.repo import get_dataset

        context = get_dataset(self.trained_on).contexts
        sources = [self.weights, context]
        target = self.responses
        if should_make(target, sources):
            self._do_sample(context)
        else:
            logger.info('sample output is up to date')

    def _do_sample(self, context):
        template = load_template('serban_sample')
        cmd = template.format(
            name=self._get_docker_name('sample'),
            model_prefix=self._get_model_prefix(),
            output=str(self.responses),
            gpu=get_random_gpu(),
            context=context,
        )
        return subprocess.check_call(cmd, shell=True)
