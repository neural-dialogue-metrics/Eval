import json
import shutil
from pathlib import Path

from corr.utils import find_all_data_files, UtterScoreDist
from pandas import DataFrame
from pylatex import Tabular, Table, Document, Command, Label, Marker, MultiColumn
from corr.normalize import normalize_name


class SystemScoreTable(Table):
    def __init__(self, caption, label, models, metrics, datasets, **kwargs):
        super().__init__(**kwargs)
        self.append(Command('centering'))
        self.add_caption(caption)
        self.append(Label(Marker(name=label, prefix='tab')))
        self.models = models
        self.datasets = datasets
        self.metrics = metrics
        self.num_columns = len(self.datasets) * len(self.models) + 1
        self.tabular = Tabular(table_spec='c' * self.num_columns)

        self.tabular.add_hline()
        self.tabular.add_row(self._dataset_header())
        self.tabular.add_row(self._model_header())

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

    def _add_scores(self):
        pass


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


if __name__ == '__main__':
    df = load_system_score(
        prefix=Path('/home/cgsdfc/Metrics/Eval/data/v2/score/db')
    )
    print(df)
