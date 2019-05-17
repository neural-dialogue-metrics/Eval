import json
import shutil
from itertools import product
from pathlib import Path
import logging
from corr.utils import find_all_data_files, UtterScoreDist
from numpy import NaN
from pandas import DataFrame
from pylatex import Tabular, Table, Document, Command, Label, Marker, MultiColumn, TableRowSizeError
from corr.normalize import normalize_name

logger = logging.getLogger(__name__)


class SystemScoreTable(Table):
    places = 4
    order = ['metric', 'dataset', 'model']

    def __init__(self, caption, label, system_scores: DataFrame, places=None, **kwargs):
        super().__init__(**kwargs)
        if places is not None:
            self.places = places

        self.append(Command('centering'))
        self.add_caption(caption)
        self.append(Label(Marker(name=label, prefix='tab')))

        self.models = sorted(system_scores.model.unique())
        self.datasets = sorted(system_scores.dataset.unique())
        self.metrics = sorted(system_scores.metric.unique())

        self.num_columns = len(self.datasets) * len(self.models) + 1
        self.tabular = Tabular(table_spec='c' * self.num_columns)

        self.tabular.add_hline()
        self.tabular.add_row(self._dataset_header())
        self.tabular.add_row(self._model_header())
        self._add_scores(system_scores)
        self.append(self.tabular)

    def _dataset_header(self):
        header = ['']
        for ds in self.datasets:
            header.append(MultiColumn(len(self.models), data=normalize_name('dataset', ds)))
        return header

    def _model_header(self):
        header = ['']
        model_names = [normalize_name('model', model) for model in self.models]
        for _ in self.datasets:
            header.extend(model_names)
        return header

    def _add_scores(self, system_scores: DataFrame):
        system_scores = self.preprocess(system_scores)
        # system_scores = self._fix_missing(system_scores)

        for metric, df in system_scores.groupby('metric'):
            row = [normalize_name('metric', metric)]
            row.extend(df.system.values)
            try:
                self.tabular.add_row(row)
            except TableRowSizeError:
                logger.error('missing value for metric {}'.format(metric))
            else:
                self.tabular.add_hline()

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


def load_system_score(prefix: Path):
    records = [json.load(file.open('r')) for file in prefix.rglob('*.json')]
    for data in records:
        del data['utterance']
    return DataFrame.from_records(records)


def new_url_schema(src_prefix: Path, dst_prefix: Path):
    # data/v2/score/db/<dataset>/<model>/<metric>/
    if not dst_prefix.exists():
        dst_prefix.mkdir(parents=True)
    files = find_all_data_files(src_prefix)
    data = [(file, UtterScoreDist.from_json_file(file)) for file in files]
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
        label='system_scores_all',
        system_scores=df,
    )
    print(table.dumps())


if __name__ == '__main__':
    make_table()
