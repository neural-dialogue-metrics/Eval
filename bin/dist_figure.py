import math
from pathlib import Path

from pandas import DataFrame
from pylatex import Figure, SubFigure, NewLine, NoEscape, Command, Label, Marker

from corr.consts import PLOT_FILENAME
from corr.normalize import normalize_name


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
    df = df[(df.model != 'random') & (df.metric != 'serban_ppl')]
    df = df.sort_values(by=['metric', 'model', 'dataset'])
    return df


def make_figures(df: DataFrame, prefix: Path):
    for metric, df2 in df.groupby('metric'):
        output = get_output(prefix, metric)
        metric_norm = normalize_name('metric', metric)
        caption = '{} 在所有模型和数据集上的概率分布'.format(metric_norm)
        label = '{}-dist-all'.format(metric_norm)
        images = df2.filename.values
        n = int(math.sqrt(len(images)))
        if pow(n, 2) != len(images):
            raise ValueError('{} != {}**2'.format(len(images), pow(n, 2)))
        figure = make_subfigures(
            images=images,
            caption=caption,
            label=label,
            n=n,
        )
        if not output.parent.exists():
            output.parent.mkdir(parents=True)
        output.write_text(figure.dumps())


def make_subfigures(images, caption='', label='', n=3):
    width = 1 / n
    fig = Figure(position='H')
    fig.append(Command('centering'))
    image_iter = iter(images)

    for i in range(n):
        for j in range(n):
            subfig = SubFigure(width=NoEscape(f'{width}' + r'\linewidth'))
            subfig.append(Command('centering'))
            image = str(next(image_iter))
            subfig.add_image(image)
            if i != 0 and j == 0:
                fig.append(NewLine())
            fig.append(subfig)

    fig.add_caption(caption)
    fig.append(Label(Marker(prefix='fig', name=label)))
    return fig


if __name__ == '__main__':
    src_prefix = Path('/home/cgsdfc/Metrics/Eval/data/v2/')
    dst_prefix = Path('/home/cgsdfc/GraduateDesign/figure')

    df = load_distplot_data(src_prefix / 'plot' / 'distplot')
    make_figures(
        prefix=dst_prefix,
        df=df,
    )
