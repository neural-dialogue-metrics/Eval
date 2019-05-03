from eval.repo import get_dataset, get_model

config = {
    'metrics': {
        'embedding_based': {
            'variants': [
                'greedy_matching',
                'vector_extrema',
                'vector_average',
            ]
        }
    },
    'models': [
        get_model('vhred', 'ubuntu'),
        get_model('lstm', 'ubuntu'),
    ],
    'datasets': [
        get_dataset('ubuntu'),
    ],
}
