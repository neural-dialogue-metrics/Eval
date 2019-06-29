"""
Correlation heatmap of various metrics on per dataset-model.
"""

from seaborn import heatmap
from iconip.utterance import load_model_dataset2_feature
from eval.data import seaborn_setup
import matplotlib.pyplot as plt

if __name__ == '__main__':
    seaborn_setup()
    model_dataset2feature = load_model_dataset2_feature()
    test = model_dataset2feature['vhred', 'opensub']
    pearson = test.corr()
    heatmap(pearson)
    plt.show()

