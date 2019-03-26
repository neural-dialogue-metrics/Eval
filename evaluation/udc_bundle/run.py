from evaluation.udc_bundle.estimator import Estimator
from evaluation.udc_bundle.loader import Loader
from evaluation.udc_bundle.model import discover_models
import evaluation.metric.builtin as m

ROUGE_N_MAX = 4
BLEU_N_MAX = 4
DISTINCT_N_MAX = 2

DRY_RUN = False


def _make_metrics():
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
        metrics.append(m.BleuScore(max_order=n, smooth=False))
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
