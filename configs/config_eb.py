from eval.repo import get_dataset, get_model
from eval.config import models, datasets

config = {
    'metrics': {
        'embedding_based': {
            'variants': [
                'greedy_matching'
                'vector_extrema',
                'vector_average',
            ]
        }
    },
    'models': [
        get_model('lstm', 'ubuntu'),
        get_model('')
    ],
    'datasets': datasets,
}
