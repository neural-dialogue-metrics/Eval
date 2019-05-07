analyzer_classes = {}


def register(cls):
    analyzer_classes[cls.name] = cls
    return cls


class Analyzer:
    name = None



