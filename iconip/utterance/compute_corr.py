from iconip.utterance import load_model_dataset2_feature


def compute_corr():
    feature = load_model_dataset2_feature()
    for key, value in feature.items():
        logger.info('computing {}-{}'.format(*key))
        for method in ['pearson', 'spearman', 'kendall']:
            logger.info('computing {}'.format(method))
            output = SAVE_ROOT / 'corr' / method / key[0] / key[1] / 'corr.json'
            output.parent.mkdir(exist_ok=True, parents=True)
            corr = value.corr(method=method)
            corr.to_json(output)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    compute_corr()
