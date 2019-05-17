import logging
from pathlib import Path

from eval.config_parser import product_models_datasets, parse_dataset
from eval.repo import all_datasets, get_dataset
from models import find_serban_models
from eval.utils import load_template, get_random_gpu
from eval.models import SerbanModel

logger = logging.getLogger(__name__)
train_template = load_template('serban_train')
sample_template = load_template('serban_sample')


def get_train(model: SerbanModel, name):
    save_dir = model.weights.parent
    model_prefix = model.weights.name.replace('_model.npz', '')
    prototype = model.prototype
    return train_template.format(
        name=name,
        save_dir=save_dir,
        prototype=prototype,
        model_prefix=model_prefix,
        gpu=get_random_gpu(),
    )


def get_sample(model: SerbanModel, name):
    model_prefix = model.weights.with_name(model.weights.name.replace('_model.npz', ''))
    context = get_dataset(model.trained_on).contexts
    output = model.responses
    return sample_template.format(
        name=name,
        model_prefix=model_prefix,
        context=context,
        output=output,
        gpu=get_random_gpu(),
    )


def train_and_sample_scripts(output_dir: Path):
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    scripts_map = {
        'train': get_train,
        'sample': get_sample,
    }

    for model in find_serban_models():
        for name, gen_fn in scripts_map.items():
            script = output_dir.joinpath('_'.join((model.name, model.trained_on, name))).with_suffix('.sh')
            logger.info('new script: {}'.format(script))
            script.write_text(gen_fn(model, script.stem))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix')
    args = parser.parse_args()
    train_and_sample_scripts(Path(args.prefix))
