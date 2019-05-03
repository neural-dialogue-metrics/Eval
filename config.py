from eval.repo import get_dataset, get_model

config = {
    'metrics': {
        'embedding_based': {
            'variants': [
                'greedy_matching'
            ]
        }
    },
    'models': [
        get_model('hred', 'ubuntu'),
    ],
    'datasets': [
        get_dataset('ubuntu'),
    ],
}
