import seaborn as sns
import matplotlib.pyplot as plt

from corr.utils import Triple

plotter_classes = {}


def register_plotter(cls):
    plotter_classes[cls.name] = cls
    return cls


class Plotter:
    name = None
    url_pattern = None

    @classmethod
    def parse_url(cls, url, loader):
        raise NotImplementedError

    def get_urls(self, config, data_index):
        raise NotImplementedError

    def reverse(self, url):
        # Return triples from a url.
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


@register_plotter
class DistPlotter(Plotter):
    name = 'distplot'
    url_pattern = '<plot>/<dataset>/<model>/<metric>/'

    def __init__(self):
        self.params = None

    def __call__(self, utterance_score):
        sns.distplot(
            a=utterance_score,
            **self.params,
        )

    @classmethod
    def parse_url(cls, url, loader):
        plot, dataset, model, metric = url.split('/')
        return {
            'utterance': loader.get_utterance_score_for(model, dataset, metric)
        }

    def get_urls(self, config, data_index):
        pass

    def reverse(self, url):
        plot, dataset, model, metric = url.split('/')
        yield Triple(model, dataset, metric)


@register_plotter
class JointPlotter(Plotter):
    name = 'jointplot'
    url_pattern = (
        '<plot>/<dataset>/model/<model>/<metric_1>-<metric_2>/',
        '<plot>/<dataset>/metric/<metric>/<model_1>-<model_2>/'
    )

    def __init__(self):
        pass

    def reverse(self, url):
        plot, dataset, mode, mode_arg, pair = url.split('/')
        pair = pair.split('-')
        if mode == 'model':
            yield (
                Triple(mode_arg, dataset, pair[0]),
                Triple(mode_arg, dataset, pair[1])
            )
        elif mode == 'metric':
            yield (
                Triple(pair[0], dataset, mode_arg),
                Triple(pair[1], dataset, mode_arg)
            )
        else:
            raise ValueError('unknown mode {}'.format(mode))

    @classmethod
    def parse_url(cls, url, loader):
        plot, dataset, mode, mode_arg, pair = url.split('/')
        pair = pair.split('-')
        if mode == 'model':
            return dict(
                label='metric',
                x=loader.get_utterance_score_for(mode_arg, dataset, pair[0]),
                y=loader.get_utterance_score_for(mode_arg, dataset, pair[1])
            )
        elif mode == 'metric':
            return dict(
                label='model',
                x=loader.get_utterance_score_for(pair[0], dataset, mode_arg),
                y=loader.get_utterance_score_for(pair[1], dataset, mode_arg)
            )
        else:
            raise ValueError('unknown mode {}'.format(mode))

    def get_urls(self, config, data_index):
        pass

    def __call__(self, x, y, label):
        grid = sns.jointplot(x=x, y=y)
        return grid


@register_plotter
class PairPlotter(Plotter):
    name = 'pairplot'
    url_pattern = (
        '<plot>/<dataset>/model/<model>/all/',
        '<plot>/<dataset>/metric/<metric>/all/',
    )

    def __init__(self, gid, groups):
        self.gid = gid
        self.groups = groups

    def reverse(self, url):
        plot, dataset, mode, mode_arg, gid = url.split('/')
        groups = self.groups
        for item in groups:
            if mode == 'model':
                yield Triple(mode_arg, dataset, item)
            elif mode == 'metric':
                yield Triple(item, dataset, mode_arg)

    @classmethod
    def parse_url(cls, url, loader):
        plot, dataset, mode, mode_arg, gid = url.split('/')

    def get_urls(self, config, data_index):
        pass

    def __call__(self, score_list, mode):
        pass
