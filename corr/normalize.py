import functools

from pandas import DataFrame


class NameNormalizationError(Exception):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f'unable to normalize {self.name}'


def upper_plus(name):
    return name.replace('_', '-').upper()


def capitalize_plus(name):
    return name.replace('_', '-').capitalize()


def normalize_eb(name):
    short_names = ('greedy', 'average', 'extrema')
    for short_name in short_names:
        if short_name in name:
            return short_name.capitalize()

    raise NameNormalizationError(name)


default_normalizers = {
    'model': str.upper,
    'dataset': {
        'ubuntu': str.capitalize,
        'opensub': 'OpenSubtitles',
        'lsdscc': str.upper,
    },
    'metric': {
        'serban_ppl': 'PPL',
        'rouge': upper_plus,
        'bleu': upper_plus,
        'meteor': upper_plus,
        'embedding_based': normalize_eb,
        'adem': str.upper,
        'distinct': capitalize_plus,
        'utterance_len': '#words',
    }
}


@functools.lru_cache()
def normalize_name(kind, name):
    normalizer = default_normalizers[kind]
    if callable(normalizer):
        return normalizer(name)
    for key, sub_normalizer in normalizer.items():
        if key not in name:
            continue
        if callable(sub_normalizer):
            return sub_normalizer(name)
        return sub_normalizer
    raise NameNormalizationError(name)


def normalize_names_in_df(df: DataFrame, names=None):
    names = names or default_normalizers.keys()
    for name in names:
        values = df[name].apply(lambda n: normalize_name(name, n))
        df[name] = values
    return df
