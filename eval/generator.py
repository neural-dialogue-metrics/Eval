import logging
from pathlib import Path

from eval.config_parser import product_models_datasets, parse_dataset
from eval.repo import find_serban_models, all_datasets
from eval.utils import load_template, get_random_gpu

dataset_out_rules = {
    'opensub': 'opensubtitles',
}

logger = logging.getLogger(__name__)
train_template = load_template('serban_train')
sample_template = load_template('serban_sample')


# prototype_opensubtitles_VHRED
def get_prototype(model, dataset):
    dataset_name = dataset_out_rules.get(dataset.name, dataset.name.lower())
    return f'prototype_{dataset_name}_{model.name.upper()}'


def get_train(model, dataset, name):
    save_dir = model.weights.parent
    model_prefix = model.weights.name.replace('_model.npz', '')
    prototype = get_prototype(model, dataset)
    return train_template.format(
        name=name,
        save_dir=save_dir,
        prototype=prototype,
        model_prefix=model_prefix,
        gpu=get_random_gpu(),
    )


def get_sample(model, dataset, name):
    model_prefix = model.weights.with_name(model.weights.name.replace('_model.npz', ''))
    context = dataset.contexts
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

    datasets = parse_dataset(all_datasets)
    models = find_serban_models()

    for model, dataset in product_models_datasets(datasets, models):
        for name, gen_fn in scripts_map.items():
            script = output_dir.joinpath('_'.join((model.name, dataset.name, name))).with_suffix('.sh')
            logger.info('new script: {}'.format(script))
            script.write_text(gen_fn(model, dataset, script.stem))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix')
    args = parser.parse_args()
    train_and_sample_scripts(Path(args.prefix))
