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
