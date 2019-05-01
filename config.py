import os
from pathlib import Path

UBUNTU_REF = '/home/cgsdfc/UbuntuDialogueCorpus/ResponseContextPairs/raw_testing_responses.txt'

GOOGLE_NEWS_300_BIN = '/home/cgsdfc/embeddings/word2vec/GoogleNews_negative300/GoogleNews-vectors-negative300.bin'


def ruber_data(train_dir, data_dir, embedding):
    data_dir = Path(data_dir)
    query_vocab = data_dir.glob('*_contexts.*.vocab*')
    query_embed = data_dir.glob('*_contexts.*.embed')
    reply_vocab = data_dir.glob('*_responses.*.vocab*')
    reply_embed = data_dir.glob('*_responses.*.embed')
    return {
        'train_dir': train_dir,
        'query_vocab': query_vocab,
        'query_embed': query_embed,
        'reply_vocab': reply_vocab,
        'reply_embed': reply_embed,
        'embedding': embedding,
    }


def data_path(response):
    parts = os.path.split(response)
    assert parts[-1].endswith('.txt'), 'path not pointing to valid output.txt'
    dataset, model = parts[-3:-1]
    return {
        'dataset': dataset.lower(),
        'model': model.lower(),
        'output': response,
    }


config = {
    'models': [
        data_path('/home/cgsdfc/Result/HRED-VHRED/Ubuntu/VHRED/output.txt'),
        data_path('/home/cgsdfc/Result/HRED-VHRED/Ubuntu/LSTM/output.txt'),
        data_path('/home/cgsdfc/Result/HRED-VHRED/Ubuntu/HRED/output.txt'),

        data_path('/home/cgsdfc/Result/HRED-VHRED/Opensub/HRED/output.txt'),
        data_path('/home/cgsdfc/Result/HRED-VHRED/Opensub/LSTM/output.txt'),
        data_path('/home/cgsdfc/Result/HRED-VHRED/Opensub/VHRED/output.txt'),
    ],
    'dataset': {
        'ubuntu': {
            'context': '/home/cgsdfc/UbuntuDialogueCorpus/ResponseContextPairs/raw_testing_contexts.txt',
            'reference': UBUNTU_REF,
        },
        'opensub3_6': {
            'context': '/home/cgsdfc/SerbanOpenSubData/dialogue_length3_6/test.context.txt',
            'reference': '/home/cgsdfc/SerbanOpenSubData/dialogue_length3_6/test.response.txt',
        }
    },
    'metrics': {
        'bleu': {
            'n': [4],
            'smoothing': False,
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
            'embeddings': {
                'file': GOOGLE_NEWS_300_BIN,
                'format': ['binary', 'word2vec'],
            }
        },
        'adem': True,
        'ruber': {
            'variants': ['hybrid', 'ref', 'unref'],
            'ubuntu': ruber_data(
                train_dir='',
                data_dir='',
                embedding='',
            )
        },
        'lsdscc': {
            'fields': ['max_bleu', 'mds', 'pds']
        }
    }
}
