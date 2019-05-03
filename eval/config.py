from eval.utils import data_path
from pathlib import Path

GOOGLE_NEWS_300_BIN = '/home/cgsdfc/embeddings/word2vec/GoogleNews_negative300/GoogleNews-vectors-negative300.bin'

output_dir = Path('/home/cgsdfc/Result/Score')

models = [
    data_path('/home/cgsdfc/Result/HRED-VHRED/Ubuntu/VHRED/output.txt'),
    data_path('/home/cgsdfc/Result/HRED-VHRED/Ubuntu/LSTM/output.txt'),
    data_path('/home/cgsdfc/Result/HRED-VHRED/Ubuntu/HRED/output.txt'),
    data_path('/home/cgsdfc/Result/HRED-VHRED/Opensub/HRED/output.txt'),
    data_path('/home/cgsdfc/Result/HRED-VHRED/Opensub/LSTM/output.txt'),
    data_path('/home/cgsdfc/Result/HRED-VHRED/Opensub/VHRED/output.txt'),
]

datasets = {
    'ubuntu': {
        'context': '/home/cgsdfc/UbuntuDialogueCorpus/ResponseContextPairs/raw_testing_contexts.txt',
        'reference': '/home/cgsdfc/UbuntuDialogueCorpus/ResponseContextPairs/raw_testing_responses.txt',
    },
    'opensub': {
        'context': '/home/cgsdfc/SerbanOpenSubData/dialogue_length3_6/test.context.txt',
        'reference': '/home/cgsdfc/SerbanOpenSubData/dialogue_length3_6/test.response.txt',
    }
}

config = {
    'models': models,
    'datasets': datasets,
    'metrics': {
        'bleu': {
            'n': [4],
            'smoothing': False,
            'filter': []
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
            'embeddings': GOOGLE_NEWS_300_BIN,
        },
    }
}
