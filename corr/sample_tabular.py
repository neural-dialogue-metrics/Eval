import json
import logging
from pathlib import Path

from eval.utils import make_parent_dirs
from pylatex import Tabularx
from pylatex.utils import bold

column2text = {
    'context': '消息：',
    'reference': '参考：',
    'response': '响应：',
}


class PieceTabular(Tabularx):
    _latex_name = 'tabularx'

    def __init__(self, data: list, **kwargs):
        super().__init__(table_spec='rX', booktabs=True, **kwargs)
        self.add_hline()
        for item in data:
            for key, value in column2text.items():
                cells = [bold(value), item[key]]
                self.add_row(cells)
            self.add_hline()


def make_tabular(src_prefix: Path = None, dst_prefix: Path = None):
    if src_prefix is None:
        src_prefix = Path('/home/cgsdfc/Metrics/Eval/data/v2/example/sample')
    if dst_prefix is None:
        dst_prefix = Path('/home/cgsdfc/GraduateDesign/data/sample')

    def get_output(path: Path):
        k, dataset, model, file = path.parts[-4:]
        output = dst_prefix / dataset / model / 'tabular.tex'
        return make_parent_dirs(output)

    for src in src_prefix.rglob('*.json'):
        data = json.load(src.open())
        tabular = PieceTabular(data)
        output = get_output(src)
        logging.info('writing tabular to {}'.format(output))
        output.write_text(tabular.dumps())


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    make_tabular()
