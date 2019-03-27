"""
Driver & Config script to conduct the experiment described in __init__.py.
"""
from evaluation.udc_bundle.estimator import Estimator
from evaluation.udc_bundle.loader import Loader
from evaluation.udc_bundle.model import discover_models
import evaluation.metric.builtin as m

# Run all variants with N <= MAX.
ROUGE_N_MAX = 4
BLEU_N_MAX = 4
DISTINCT_N_MAX = 2

BLEU_SMOOTH = False
DRY_RUN = False


def _make_metrics():
    """
    Create the metric array we want to run.

    :return:
    """
    metrics = [
        m.AverageScore(),
        m.GreedyMatchingScore(),
        m.ExtremaScore(),
        m.RougeL(),
        m.RougeW(),
    ]
    for n in range(1, ROUGE_N_MAX + 1):
        metrics.append(m.RougeN(n))
    for n in range(1, BLEU_N_MAX + 1):
        metrics.append(m.BleuScore(max_order=n, smooth=BLEU_SMOOTH))
    for n in range(1, DISTINCT_N_MAX + 1):
        metrics.append(m.DistinctN(n))
    return metrics


if __name__ == '__main__':
    estimator = Estimator(Loader.create_for_udc(), dry_run=DRY_RUN)
    for metric in _make_metrics():
        estimator.add_metric(metric)

    for model in discover_models():
        estimator.add_model(model)

    estimator.run()
