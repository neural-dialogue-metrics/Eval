import itertools
import logging
import pprint

from eval.config_parser import parse_models_and_datasets, parse_metrics
from eval.exporter import Exporter
from eval.loader import ResourceLoader
from eval.utils import UnderTest

logger = logging.getLogger(__name__)


class Engine:
    def __init__(self, config, save_dir):
        self.exporter = Exporter(save_dir)
        self.loader = ResourceLoader()
        self.config = config
        try:
            from eval.metrics import metrics_classes
            self.metrics_classes = metrics_classes
        except ImportError:
            logger.error('metric_classes not available. Some of the packages were not installed?')
            raise
        self.under_tests = self.parse_config(config)

    def parse_config(self, config):
        metrics = parse_metrics(config['metrics'], self.metrics_classes)
        models_and_datasets = parse_models_and_datasets(config)
        return [
            UnderTest(metric=metric, model=model, dataset=dataset)
            for metric, (model, dataset) in itertools.product(metrics, models_and_datasets)
        ]

    def run(self):
        logger.info('save_dir: %s', self.exporter.save_dir)
        logger.info('config: %s', pprint.pformat(self.config))

        self.exporter.export_config(self.config)
        for under_test in self.under_tests:
            logger.info('Model: %s, Dataset: %s, Metric: %s', under_test.model_name,
                        under_test.dataset_name, under_test.metric_name)
            logger.info('Loading resources: %s', ', '.join(under_test.metric.requires))
            payload = {key: self.loader.load(key, under_test) for key in under_test.metric.requires}

            logger.info('Calculating scores...')
            result = under_test.metric(**payload)

            self.exporter.export_json(result, under_test)