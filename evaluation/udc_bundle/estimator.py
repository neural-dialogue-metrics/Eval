"""
This module handles the process of evaluating various metrics on various models.
"""
import collections
import logging
import pathlib
import tempfile
import multiprocessing

from evaluation.udc_bundle.model import ModelInfo

_logger = logging.getLogger(__name__)

_MODEL_METRIC_FILE = '%(model)s-%(metric)s.txt'
_SUMMARY_FILE = 'summary.csv'
_CSV_HEADER = 'Metric,Model,Dataset,Value'
_DATASET = 'UDC'


def _schedule_metrics(metrics):
    """
    First sort the list of metrics to let those requiring less resources to run first.
    Then assuming the metrics from the same class will cost nearly the same amount of time,
    group them into a list and try to run them in parallel.

    :param metrics:
    :return:
    """
    _metrics = collections.defaultdict(list)
    # Bleu-2 and Bleu-3 will be in one group but ROUGE-2 and ROUGE-N won't be in one group.
    # This is just a hint.
    for m in metrics:
        _metrics[type(m)] = m
    return sorted(_metrics.items(), key=lambda ms: len(ms[1][0].signature))


class Estimator:
    """
    This class controls the whole process of evaluation.
    """

    def __init__(self, loader, dry_run=False):
        """

        :param loader: a Loader object that can load a resource.
        See Loader.load().

        :param dry_run: bool. Don't create any files when true.
        """
        self._dry_run = dry_run
        self._models = []
        self._metrics = []
        self._results = []
        self._loader = loader
        self._pool = multiprocessing.Pool(processes=2)
        if not self._dry_run:
            self._out_dir = pathlib.Path(tempfile.mkdtemp())
            _logger.info('our_dir: %s', self._out_dir)

    def add_metric(self, metric):
        _logger.info('added metric %s', metric)
        self._metrics.append(metric)

    def add_model(self, model):
        _logger.info('added model %s', model)
        self._models.append(model)

    def _load_signature(self, signature, model: ModelInfo):
        """
        Load signature for a metric.

        :param signature:
        :return:
        """
        return {key: self._loader.load(key, model.first_response_path) for key in signature}

    def _run_metric_group(self, metrics, **kwargs):
        """
        Run a group of metrics.

        :param metrics: a list of metrics sharing the same class.
        :param kwargs: loaded signature.
        :return: a list of result values.
        """
        results = []
        for metric in metrics:
            _logger.info('apply metric %s...', metric)
            future = self._pool.apply_async(func=metrics, kwds=kwargs)
            results.append(future)
        _results = []
        for metric, future in zip(metrics, results):
            _logger.info('waiting for metric %s', metrics)
            r = future.get()
            _logger.info('done. metric %s: %r', metric, r)
            _results.append(r)
        return _results

    def _dump_result(self, model, metrics, results):
        """
        Save the result of running metrics on a model to a file.

        :param model: ModelInfo.
        :param metrics: a list of Metrics.
        :param results: a list what a metric() returns.
        :return:
        """
        for metric, result in zip(metrics, results):
            filename = self._out_dir / _MODEL_METRIC_FILE % dict(model=model, metric=metric)
            _logger.info('writing to %s', filename)
            with open(filename, 'w') as f:
                print(metric.to_scalar(result), file=f)
            self._results.append((model, metric, filename))

    def _write_summary(self, filename):
        _logger.info('writing summary to %s...', filename)
        with open(filename, 'w') as f:
            print(_CSV_HEADER, file=f)
            for model, metric, result_file in self._results:
                with open(result_file) as rf:
                    value = rf.read().strip()
                print(metric, model, _DATASET, value, sep=',', file=f)

    def run(self):
        _logger.info('scheduling metrics...')
        self._metrics = _schedule_metrics(self._metrics)

        for model in self._models:
            for metric_class, metric_group in self._metrics:
                _logger.info('loading signature...')
                kwargs = self._load_signature(metric_class.signature, model)
                _logger.info('evaluating %s on %s...', metric_class.__name__, model)
                results = self._run_metric_group(metric_group, **kwargs)
                if not self._dry_run:
                    self._dump_result(model, metric_group, results)

        _logger.info('all evaluation done')
        if not self._dry_run:
            self._write_summary(self._out_dir / _SUMMARY_FILE)
