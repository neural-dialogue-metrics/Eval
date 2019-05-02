from eval.utils import data_path, ruber_data

GOOGLE_NEWS_300_BIN = '/home/cgsdfc/embeddings/word2vec/GoogleNews_negative300/GoogleNews-vectors-negative300.bin'

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
            'reference': '/home/cgsdfc/UbuntuDialogueCorpus/ResponseContextPairs/raw_testing_responses.txt',
        },
        'opensub': {
            'context': '/home/cgsdfc/SerbanOpenSubData/dialogue_length3_6/test.context.txt',
            'reference': '/home/cgsdfc/SerbanOpenSubData/dialogue_length3_6/test.response.txt',
        }
    },
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
