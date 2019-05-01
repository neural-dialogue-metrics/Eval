import os
from pathlib import Path

UBUNTU_CONTEXT = '/home/cgsdfc/UbuntuDialogueCorpus/ResponseContextPairs/raw_testing_contexts.txt'
UBUNTU_REF = '/home/cgsdfc/UbuntuDialogueCorpus/ResponseContextPairs/raw_testing_responses.txt'

OPENSUB_CTX = '/home/cgsdfc/SerbanOpenSubData/dialogue_length3_6/test.context.txt'
OPENSUB_REF = '/home/cgsdfc/SerbanOpenSubData/dialogue_length3_6/test.response.txt'

GOOGLE_NEWS_300_BIN = '/home/cgsdfc/embeddings/word2vec/GoogleNews_negative300/GoogleNews-vectors-negative300.bin'


def data_path(response, context=None, reference=None):
    parts = os.path.split(response)
    assert parts[-1].endswith('.txt'), 'path not pointing to valid output.txt'
    dataset, model = parts[-3:-1]
    return {
        'dataset': dataset.lower(),
        'model': model.lower(),
        'response': response,
        'context': context,
        'reference': reference,
    }


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


config = {
    'under_test': [
        data_path('/home/cgsdfc/Result/HRED-VHRED/Ubuntu/VHRED/output.txt'),
        data_path('/home/cgsdfc/Result/HRED-VHRED/Ubuntu/LSTM/output.txt'),
        data_path('/home/cgsdfc/Result/HRED-VHRED/Ubuntu/HRED/output.txt'),

        data_path('/home/cgsdfc/Result/HRED-VHRED/Opensub/HRED/output.txt'),
        data_path('/home/cgsdfc/Result/HRED-VHRED/Opensub/LSTM/output.txt'),
        data_path('/home/cgsdfc/Result/HRED-VHRED/Opensub/VHRED/output.txt'),
    ],
    'dataset': {
        'ubuntu': {
            'context': UBUNTU_CONTEXT,
            'reference': UBUNTU_REF,
        },
        'opensub': {
            'context': OPENSUB_CTX,
            'reference': OPENSUB_REF,
        }
    },
    'metrics': {
        'bleu': {
            'n': [4],
            'smoothing': False,
        },
        'rouge': {
            'alpha': 'default',
            'rouge_n': [4],
            'rouge_l': True,
            'rouge_w': True,
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
            'variants': ['max_bleu', 'mds', 'pds']
        }
    }
}
