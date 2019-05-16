import argparse
from pathlib import Path
from typing import Union, List

from eval.consts import SEPARATOR, CONTEXTS, REFERENCES, RESPONSES
from eval.loader import ResourceLoader
from eval.utils import Model, UnderTest
from eval.repo import all_models, get_dataset
from pandas import DataFrame
import logging

logger = logging.getLogger(__name__)


class ExampleMaker:
    supported_formats = ('csv', 'json')
    attrs = ('contexts', 'references', 'responses')
    requires = {
        REFERENCES: 'lines',
        CONTEXTS: 'lines',
        RESPONSES: 'lines',
    }

    def __init__(self, n_examples: int, prefix: Union[str, Path],
                 output_format: str, random_state: int = None):
        self.n_examples = n_examples
        self.prefix = Path(prefix)
        self.loader = ResourceLoader()
        self.random_state = random_state
        if output_format in self.supported_formats:
            self.output_format = output_format
        else:
            raise ValueError('unsupported output_format {}'.format(output_format))

    def load_model_examples(self, model: Model):
        data = self.loader.load_resources(
            UnderTest(
                metric=self,
                model=model,
                dataset=get_dataset(model.trained_on),
            )
        )
        return DataFrame(data)

    def sample(self, df: DataFrame):
        return df.sample(n=self.n_examples, random_state=self.random_state)

    def get_output_path(self, model: Model):
        suffix = '.' + self.output_format
        return self.prefix.joinpath(
            SEPARATOR.join((model.name, model.trained_on))).with_suffix(suffix)

    def get_save_fn(self, df):
        return getattr(df, 'to_' + self.output_format)

    def make_example_for(self, model: Model):
        logger.info('loading files for model {}'.format(model))
        df = self.load_model_examples(model)

        logger.info('sampling examples...')
        df = self.sample(df)

        output = self.get_output_path(model)
        if not output.parent.exists():
            output.parent.mkdir(parents=True)
        logger.info('saving to {}'.format(output))
        save_fn = self.get_save_fn(df)
        save_fn(output)

    def make_examples(self, models: List[Model]):
        for model in models:
            self.make_example_for(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Make examples by randomly sampling')
    parser.add_argument('-p', '--prefix', help='output prefix of the example files')
    parser.add_argument('-n', '--n-examples', type=int, default=15, help='number of examples to draw')
    parser.add_argument('-f', '--format', help='output format of an individual example',
                        choices=ExampleMaker.supported_formats)
    parser.add_argument('-s', '--seed', type=int, default=123,
                        help='random number seed that helps to reproduce the results')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    maker = ExampleMaker(
        n_examples=args.n_examples,
        prefix=args.prefix,
        output_format=args.format,
        random_state=args.seed,
    )

    maker.make_examples(all_models)
