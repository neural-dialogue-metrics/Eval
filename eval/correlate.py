import json
import pandas as pd
from sklearn.preprocessing import scale as sklearn_scale
from eval.repo import get_model, get_dataset
from eval.consts import SAMPLE_SIZE, RANDOM_STATE
import seaborn as sns


class UtterScoreDist:
    """Utterance-Score Distribution"""

    def __init__(self, model, dataset, metric, system, utterance):
        self.model = model
        self.dataset = dataset
        self.metric = metric
        self.system = system
        self.utterance = pd.Series(data=utterance,
                                   name=f'{metric} score of {model} trained on {dataset}')

    def get_model(self):
        return get_model(self.model, self.dataset)

    def get_dataset(self):
        return get_dataset(self.dataset)

    @property
    def mean(self):
        return self.utterance.mean()

    @property
    def std(self):
        return self.utterance.std()

    @classmethod
    def from_json_file(cls, filename):
        data = json.load(open(filename))
        return cls(**data)

    def scaled_utterance(self):
        return self.utterance.transform(sklearn_scale)

    def plot_dist(self):
        data = self.utterance.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)
        data = data.transform(sklearn_scale)
        return sns.distplot(data)
