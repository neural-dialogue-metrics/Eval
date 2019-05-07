import itertools
import logging
import pprint
from pathlib import Path

from eval.config_parser import parse_models_and_datasets, parse_metrics
from eval.exporter import Exporter
from eval.loader import ResourceLoader
from eval.utils import UnderTest

logger = logging.getLogger(__name__)


def parse_config(config):
    metrics = parse_metrics(config['metrics'])
    models_and_datasets = parse_models_and_datasets(config)
    return [
        UnderTest(metric=metric, model=model, dataset=dataset)
        for metric, (model, dataset) in itertools.product(metrics, models_and_datasets)
    ]


class Engine:
    def __init__(self, config, save_dir, force=False):
        self.exporter = Exporter(save_dir)
        self.loader = ResourceLoader()
        self.config = config
        self.force = force
        self.under_tests = parse_config(config)

    def is_outdated(self, output: Path, under_test):
        if not output.exists():
            return True
        filenames = self.loader.get_filenames(under_test).values()
        for file in filenames:
            if file.stat().st_mtime > output.stat().st_mtime:
                logger.info('file {} is newer than {}'.format(file, output))
                return True
        return False

    def run(self):
        logger.info('save_dir: %s', self.exporter.save_dir)
        logger.info('config: %s', pprint.pformat(self.config))

        self.exporter.export_config(self.config)
        for under_test in self.under_tests:
            output_path = self.exporter.get_output_path(under_test)
            if not self.is_outdated(output_path, under_test) and not self.force:
                logger.info('skipping up-to-date file %s', output_path)
                continue

            logger.info('Running under_test: %r', under_test)
            payload = self.loader.load_resources(under_test)
            if payload is None:
                continue

            logger.info('Calculating scores...')
            try:
                result = under_test.metric(**payload)
            except KeyboardInterrupt:
                logging.warning('interrupted, skipping...')
            else:
                self.exporter.export_json(result, under_test)
        logger.info('all done')
