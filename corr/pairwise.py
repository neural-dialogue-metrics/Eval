import json
import logging
import pprint
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from eval.consts import PEARSON, SPEARMAN, SCATTER_ALPHA
from corr.utils import UtterScoreDist

logger = logging.getLogger(__name__)

method_map = {
    PEARSON: pearsonr,
    SPEARMAN: spearmanr,
}


def correlation(u: UtterScoreDist, v: UtterScoreDist, method=PEARSON):
    fn = method_map[method]
    return fn(u.utterance, v.utterance)


def scatter_plot(u: UtterScoreDist, v: UtterScoreDist):
    if u.dataset != v.dataset:
        raise ValueError(f'datasets of u,v must be the same. Got {u.dataset} for u and {v.dataset} for v')
    plt.scatter(u.utterance, v.utterance, alpha=SCATTER_ALPHA)
    if u.model == v.model:
        plt.title(f'{u.metric} vs {v.metric} on {u.model}-{u.dataset}')
        plt.xlabel(f'{u.metric}')
        plt.ylabel(f'{v.metric}')
    elif u.metric == v.metric:
        plt.title(f'{u.model} vs {v.model} on {u.metric}-{u.dataset}')
        plt.xlabel(f'{u.model}')
        plt.ylabel(f'{v.model}')
    else:
        raise ValueError('one of (metric, model) must be the same')


def compute_corr_and_plot_scatter(u: Path, v: Path, output_dir: Path):
    logger.info('loading score %s', u)
    u = UtterScoreDist.from_json_file(u)

    logger.info('loading score %s', v)
    v = UtterScoreDist.from_json_file(v)

    logger.info('computing correlation...')
    corr = {
        name: correlation(u, v, name) for name in (PEARSON, SPEARMAN)
    }
    logger.info('%s', pprint.pformat(corr))

    basename = output_dir.joinpath('__vs__'.join((u.name, v.name)))
    corr_output = basename.with_suffix('.json')
    payload = {
        'u': u.parts,
        'v': v.parts,
        'corr': corr,
    }

    logger.info('writing correlation to %s', corr_output)
    corr_output.write_text(json.dumps(payload))

    logger.info('plotting scatter plot...')
    scatter_plot(u, v)
    plt.savefig(basename.with_suffix('.png'))
