from pathlib import Path
from typing import List

from corr.consts import *
from corr.utils import load_filename_data, Triple
from corr.loader import ResourceLoader


class Engine:
    def __init__(self, config, data_root, output_root):
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        self.data_index = load_filename_data(self.data_root)
        self.plotters = self.create_plotters(config)
        self.urls = list(self.create_urls(config))
        self.loader = ResourceLoader(self.data_index)

    def create_urls(self, config):

        def do_parse():
            for key, plotter in self.plotters.items():
                yield from plotter.get_urls(config, self.data_index)

        return list(map(Path, do_parse()))

    def is_outdated(self, triples: List[Triple], output_file: Path):
        for path in map(self.loader.lookup_triple, triples):
            if not path.exists():
                return True
            if path.stat().st_mtime > output_file.stat().st_mtime:
                return True
        return False

    def run(self):
        for url in self.urls:
            output_file = self.output_root.joinpath(url).joinpath(OUTPUT_FILE)
            plot_key = url.parts[0]


    def create_plotters(self, config):
        plotters = {}
        from corr.plotters import plotter_classes
        for key, value in config.items():
            plotter_cls = plotter_classes[key]
            plotters[key] = plotter_cls.parse_config(value)
        return plotters
