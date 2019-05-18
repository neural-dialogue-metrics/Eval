import shutil
from itertools import product
from pathlib import Path
import logging

from pylatex.utils import bold

from corr.utils import find_all_data_files, UtterScoreDist
from pandas import DataFrame
from pylatex import Tabular, Table, Command, Label, Marker, MultiColumn, TableRowSizeError
from corr.normalize import normalize_name
from eval.data import load_system_score

__version__ = '0.0.1'

logger = logging.getLogger(__name__)


class SystemScoreTable(Table):
    _latex_name = 'table'
    places = 4
    order = ['metric', 'dataset', 'model']

    def __init__(self, caption, label, system_scores: DataFrame, places=None, **kwargs):
        super().__init__(position='H', **kwargs)
        if places is not None:
            self.places = places

        self.append(Command('centering'))
        self.add_caption(caption)
        self.append(Label(Marker(name=label, prefix='tab')))

        self.models = sorted(system_scores.model.unique())
        self.datasets = sorted(system_scores.dataset.unique())
        self.metrics = sorted(system_scores.metric.unique())

        self.num_columns = len(self.datasets) * len(self.models) + 1
        self.tabular = Tabular(table_spec='l' * self.num_columns, booktabs=True, col_space='0.11cm')

        self.tabular.add_hline()
        self.tabular.add_row(self._dataset_header())
        self.tabular.add_hline()
        self.tabular.add_row(self._model_header())
        self.tabular.add_hline()
        self._add_scores(system_scores)
        self.append(self.tabular)

    def _dataset_header(self):
        header = ['']
        for ds in self.datasets:
            data = normalize_name('dataset', ds)
            header.append(MultiColumn(len(self.models), data=data, align='c'))
        return header

    def _model_header(self):
        header = ['']
        model_names = [normalize_name('model', model) for model in self.models]
        for _ in self.datasets:
            header.extend(model_names)
        return header

    def _get_model_scores(self, df: DataFrame, metric):
        scores = list(map(str, df['system'].values))
        if metric == 'serban_ppl':
            fn = 'argmin'
        elif metric == 'utterance_len':
            return scores
        else:
            fn = 'argmax'
        pos = getattr(df['system'].values, fn)()
        scores[pos] = bold(scores[pos])
        return scores

    def _add_scores(self, system_scores: DataFrame):
        system_scores = self.preprocess(system_scores)
        # system_scores = self._fix_missing(system_scores)

        for metric, df in system_scores.groupby('metric'):
            row = [normalize_name('metric', metric)]
            for dataset, df2 in df.groupby('dataset'):
                row.extend(self._get_model_scores(df2, metric))
            try:
                self.tabular.add_row(row)
            except TableRowSizeError:
                logger.error('missing value for metric {}'.format(metric))

    def _fix_missing(self, system_scores: DataFrame):
        values = system_scores.values
        values = set((row[1], row[0], row[2]) for row in values)
        index = [getattr(self, key + 's') for key in self.order]
        missing = 0
        for arg in product(*index):
            if arg not in values:
                logger.error('{} absent'.format(arg))
                missing += 1
        if missing:
            logger.error('{} missing'.format(missing))
        return system_scores

    def preprocess(self, system_scores: DataFrame):
        system_scores = system_scores[system_scores.model != 'random']
        return system_scores.sort_values(self.order).round(self.places)


class TablePerDataset(Table):
    _latex_name = 'table'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def new_url_schema(src_prefix: Path, dst_prefix: Path):
    # data/v2/score/db/<dataset>/<model>/<metric>/
    if not dst_prefix.exists():
        dst_prefix.mkdir(parents=True)
    files = find_all_data_files(src_prefix)
    data = [(file, UtterScoreDist(file)) for file in files]
    for file, dist in data:
        dst_url = dst_prefix / dist.dataset / dist.model / dist.metric
        if not dst_url.exists():
            dst_url.mkdir(parents=True)
        shutil.copy(file, dst_url)


def migrate_url_schema():
    new_url_schema(
        src_prefix=Path('/home/cgsdfc/Metrics/Eval/data/v2/score/all'),
        dst_prefix=Path('/home/cgsdfc/Metrics/Eval/data/v2/score/db'),
    )


def make_table():
    df = load_system_score(
        prefix=Path('/home/cgsdfc/Metrics/Eval/data/v2/score/db')
    )
    df = df[df.model != 'random']
    table = SystemScoreTable(
        caption='不同数据集上的模型的各种指标得分',
        label='systemScoresAll',
        system_scores=df,
    )
    output = Path('/home/cgsdfc/GraduateDesign/data/system_scores.tex')
    output.write_text(table.dumps())


if __name__ == '__main__':
    make_table()
