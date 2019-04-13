import importlib


class Config:
    DEFAULTS = {
        'dry_run': False,
        'models': [],
        'metrics': [],
    }

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, self.__class__):
            return False
        return

    def _find_items(self, config_dict, ignore_case=False):
        out = {}
        for key, value in self.DEFAULTS.items():


    def __init__(self, **kwargs):
        for key, value in self.DEFAULTS.items():
            pass

    @classmethod
    def from_module_path(cls, mod_path):
        mod = importlib.import_module(mod_path)
        return cls(**mod.__dict__)

    @classmethod
    def from_filename(cls, filename):
        _dict = {}
        with open(filename) as f:
            exec(f.read(), _dict)
        return cls(**_dict)
