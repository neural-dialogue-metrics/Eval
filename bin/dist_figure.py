from pathlib import Path

import pandas
from pandas import DataFrame
from pylatex import Figure, SubFigure, Document, NewLine

from corr.consts import PLOT_FILENAME


def get_output(prefix: Path, metric):
    return prefix / 'latex' / 'figure' / 'distplot' / metric / 'figure.tex'


def load_distplot_data(prefix: Path):
    files = list(prefix.rglob(PLOT_FILENAME))

    def parse(url: Path):
        dataset, model, metric = url.parts[-4:-1]
        return dict(
            dataset=dataset,
            model=model,
            metric=metric,
            filename=url,
        )

    df = DataFrame.from_records(map(parse, files))
    return df[(df.model != 'random') & (df.metric != 'serban_ppl')]


PREFIX = Path('/home/cgsdfc/Metrics/Eval/data/v2/plot/distplot')


def make_subfigures(image, n=3):
    width = 1 / n
    fig = Figure()
    for i in range(n):
        for j in range(n):
            subfig = SubFigure()
            subfig.add_image(image)
            if i != 0 and j == 0:
                fig.append(NewLine())
            fig.append(subfig)
    return fig


if __name__ == '__main__':
    image = '/home/cgsdfc/Metrics/Eval/data/v2/plot/distplot/lsdscc/hred/adem/plot.pdf'
    doc = Document('subfig')
    fig = make_subfigures(image, 2)
    doc.append(fig)
    print(fig.dumps())
    doc.generate_pdf()
