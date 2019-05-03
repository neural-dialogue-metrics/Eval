from pathlib import Path

from eval.consts import REFERENCES, CONTEXTS
from eval.utils import model_path

output_dir = Path('/home/cgsdfc/Result/Score')

models = [
    model_path('/home/cgsdfc/Result/HRED-VHRED/Ubuntu/VHRED/output.txt'),
    model_path('/home/cgsdfc/Result/HRED-VHRED/Ubuntu/LSTM/output.txt'),
    model_path('/home/cgsdfc/Result/HRED-VHRED/Ubuntu/HRED/output.txt'),
    model_path('/home/cgsdfc/Result/HRED-VHRED/Opensub/HRED/output.txt'),
    model_path('/home/cgsdfc/Result/HRED-VHRED/Opensub/LSTM/output.txt'),
    model_path('/home/cgsdfc/Result/HRED-VHRED/Opensub/VHRED/output.txt'),
]

datasets = {
    'ubuntu': {
        CONTEXTS: '/home/cgsdfc/UbuntuDialogueCorpus/ResponseContextPairs/raw_testing_contexts.txt',
        REFERENCES: '/home/cgsdfc/UbuntuDialogueCorpus/ResponseContextPairs/raw_testing_responses.txt',
    },
    'opensub': {
        CONTEXTS: '/home/cgsdfc/SerbanOpenSubData/dialogue_length3_6/test.context.txt',
        REFERENCES: '/home/cgsdfc/SerbanOpenSubData/dialogue_length3_6/test.response.txt',
    }
}

metrics = {
    'bleu': {
        'n': [4],
        'smoothing': True,
    },
    'rouge': {
        'alpha': 0.9,
        'weight': 1.2,
        'n': [2],
        'variants': ['rouge_n', 'rouge_l', 'rouge_w'],
    },
    'distinct_n': {
        'n': [1, 2]
    },
    'embedding_based': {
        'variants': [
            'vector_average',
            'vector_extrema',
            'greedy_matching',
        ],
    },
}

config = {
    'models': models,
    'datasets': datasets,
    'metrics': metrics
}
