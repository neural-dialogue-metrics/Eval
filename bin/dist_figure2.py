from pathlib import Path
from corr.normalize import normalize_name

from pylatex import Figure, Label, Marker, NoEscape


def make_figures(plot_prefix: Path, dst_prefix: Path):
    files = list(plot_prefix.rglob('*.pdf'))

    for file in files:
        metric = file.parent.name
        norm_metric = normalize_name('metric', metric)
        output = dst_prefix.joinpath(metric).with_suffix('.tex')
        figure = Figure(position='H')
        figure.add_image(filename=str(file))
        figure.add_caption('{} 的概率分布图'.format(norm_metric))
        figure.append(Label(Marker(
            prefix='fig',
            name=NoEscape('{}dist'.format(norm_metric)),
        )))
        output.write_text(figure.dumps())


if __name__ == '__main__':
    plot_prefix = Path('/home/cgsdfc/Metrics/Eval/data/v2/plot/distplot_grid')
    dst_prefix = Path('/home/cgsdfc/GraduateDesign/data').joinpath('distplot_grid')
    if not dst_prefix.exists():
        dst_prefix.mkdir(parents=True)

    make_figures(
        plot_prefix=plot_prefix,
        dst_prefix=dst_prefix,
    )
