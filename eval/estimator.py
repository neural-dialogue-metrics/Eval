"""
This module handles the process of evaluating various metrics on various models.
"""
import collections
import logging
import pathlib
import tempfile
import multiprocessing


_logger = logging.getLogger(__name__)

# The filename format for saving a metric score to file for later loading.
_MODEL_METRIC_FILE = '%(model)s-%(metric)s.txt'

# The header of the summary csv file.
_CSV_HEADER = 'Metric,Model,Dataset,Value'

# The value for the Dataset column. In our case, it is all UDC.
_DATASET = 'UDC'


def _schedule_metrics(metrics):
    """
    Sort the list of metrics to let those requiring less resources to run first.
    Assuming the metrics from the same class will cost nearly the same amount of time,
    group them into a list and try to run them in parallel.
    NB: the classes of the same super class is not considered, i.e., AverageScore and ExtremaScore
    won't be in the same group.

    >>> from eval.metric.builtin import *
    >>> metrics = [RougeN(1), RougeN(2), BleuScore(3, smooth=False), AverageScore(), DistinctN(3)]
    >>> metrics = _schedule_metrics(metrics)
    >>> [(cls.__name__, ', '.join(map(str, m))) for cls, m in metrics]
    [('DistinctN', 'Distinct-3'), ('RougeN', 'ROUGE-1, ROUGE-2'), ('BleuScore', 'BLEU-3'), ('AverageScore', 'average')]

    :param metrics: a list of Metrics.
    :return: List[Tuple[type, list]].
    """
    _metrics = collections.defaultdict(list)
    for m in metrics:
        _metrics[type(m)].append(m)
    # Assuming signature is a class attribute.
    return sorted(_metrics.items(), key=lambda ms: len(ms[0].signature))


def _log_metric_schedule(metrics):
    """
    Print the metrics as a result of scheduling.

    :param metrics: returned by __schedule_metrics().
    :return:
    """
    _logger.info('Metric Schedule:')
    for cls, _metrics in metrics:
        _logger.info('%s: %s', cls.__name__, ', '.join(map(str, _metrics)))
    _logger.info('Metric Schedule End')


class Estimator(object):
    """
    This class controls the whole process of eval.
    The eval runs in a grid fashion -- the row lists the metrics and the column lists the models.
    It runs in row-major -- all metrics for a model is run, and then for the next model.
    This can be cache-friendly since files of a model is accessed consecutively and only in one period.
    """

    def __init__(self, loader, dry_run=False):
        """
        :param loader: a Loader object that can load a resource given a key in the Signature.
        See Loader.load().
        :param dry_run: bool. Don't create any file when true.
        """
        self._dry_run = dry_run
        # a list of ModelInfo.
        self._models = []
        # a list of Metric.
        self._metrics = []
        # a list of 3-tuples: (model, metric, filename), where filename save the result.
        self._results = []
        # loader loads and keeps the resource.
        self._loader = loader
        # run the metrics of the same class in parallel.
        self._pool = multiprocessing.Pool(processes=2)
        if not self._dry_run:
            # save output files (maybe a ton of number!) in the dir.
            self._out_dir = pathlib.Path(tempfile.mkdtemp())
            _logger.info('our_dir: %s', self._out_dir)

    def add_metric(self, metric):
        """
        Add a metric to the grid.

        :param metric: a Metric instance.
        :return:
        """
        _logger.info('added metric %s', metric)
        self._metrics.append(metric)

    def add_model(self, model):
        """
        Add a metric to the grid.

        :param model: a ModelInfo instance.
        :return:
        """
        _logger.info('added model %s', model)
        self._models.append(model)

    def _load_signature(self, signature, model):
        """
        Load signature for a metric.

        :param signature: found in a metric's signature attribute.
        :return: a kwargs that can be passed to a metric's __call__().
        """
        _logger.info('loading signature...')
        return {key: self._loader.load(key, model.first_response_path) for key in signature}

    def _run_metric_group(self, metrics, **kwargs):
        """
        Run a group of metrics.

        :param metrics: a list of metrics sharing the same class.
        :param kwargs: a loaded signature.
        :return: a list of result values.
        """
        if len(metrics) == 1:
            _metric = metrics[0]
            _logger.info('apply metric %s...', _metric)
            r = _metric(**kwargs)
            _logger.info('done. metric %s: %r', _metric, r)
            return [r]

        futures = []
        for _metric in metrics:
            _logger.info('apply metric %s...', _metric)
            future = self._pool.apply_async(func=_metric, kwds=kwargs)
            futures.append(future)
        _results = []
        for _metric, future in zip(metrics, futures):
            _logger.info('waiting for metric %s', _metric)
            r = future.get()
            _logger.info('done. metric %s: %r', _metric, r)
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
            basename = _MODEL_METRIC_FILE % dict(model=model, metric=metric)
            filename = self._out_dir / basename
            _logger.info('saving result to %s', filename)
            with open(filename, 'w') as f:
                print(metric.to_scalar(result), file=f)
            self._results.append((model, metric, filename))

    def _write_summary(self, filename):
        """
        Create the summary.csv file.

        :param filename:
        :return:
        """
        _logger.info('writing summary to %s...', filename)
        with open(filename, 'w') as f:
            print(_CSV_HEADER, file=f)
            for model, metric, result_file in self._results:
                with open(result_file) as rf:  # load the value from file.
                    value = rf.read().strip()
                # Be careful of the order!
                print(metric, model, _DATASET, value, sep=',', file=f)

    def run(self):
        """
        Run the grid eval.

        :return:
        """
        # print some statistics.
        _logger.info('number of models: %d', len(self._models))
        _logger.info('number of metrics: %d', len(self._metrics))
        _logger.info('number of combinations: %d', len(self._models) * len(self._metrics))

        _logger.info('scheduling metrics...')
        self._metrics = _schedule_metrics(self._metrics)
        _log_metric_schedule(self._metrics)

        for model in self._models:
            _logger.info('Evaluating Model %s', model)

            for metric_class, metric_group in self._metrics:
                _logger.info('evaluating %s on %s...', metric_class.__name__, model)
                kwargs = self._load_signature(metric_class.signature, model)
                results = self._run_metric_group(metric_group, **kwargs)
                if not self._dry_run:
                    self._dump_result(model, metric_group, results)
        _logger.info('all eval done')

        if not self._dry_run:
            self._write_summary(self._out_dir / SUMMARY_FILE)
