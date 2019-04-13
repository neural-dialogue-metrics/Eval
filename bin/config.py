"""
This package runs our existing metrics on the context-response pairs bundled
with the Ubuntu Dialogue Corpus.

Our metrics are:
    - EmbeddingBased
        * Average
        * Extrema
        * Greedy Matching
    - BLEU
        * BLEU-1
        * BLEU-2
        * BLEU-3
        * BLEU-4
    - ROUGE
        * ROUGE-1
        * ROUGE-2
        * ROUGE-L
        * ROUGE-W
    - Distinct-N
        * Distinct-1
        * Distinct-2

This module defines some common constants for other parts.
"""
import pathlib

UDC_ROOT = '/home/cgsdfc/UbuntuDialogueCorpus'

RESPONSE_CONTEXT_PAIRS = 'ResponseContextPairs'
MODEL_PREDICTIONS = 'ModelPredictions'
TESTING_RESPONSES = 'raw_testing_responses.txt'
FIRST_RESPONSE_SUFFIX = '_First.txt'

EVAL_ROOT = pathlib.Path(UDC_ROOT) / RESPONSE_CONTEXT_PAIRS
MODEL_ROOT = EVAL_ROOT / MODEL_PREDICTIONS
REFERENCE_CORPUS_PATH = EVAL_ROOT / TESTING_RESPONSES

EMBEDDING_PATH = '/home/cgsdfc/embeddings/GoogleNews-vectors-negative300.bin'

# Control which model we want to handle.
KNOWN_MODELS = (
    'HRED_Baseline',
    'LSTM_Baseline',
    'VHRED',
)

# The filename of the summary file.
SUMMARY_FILE = 'summary.csv'

METRICS = [

]

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
