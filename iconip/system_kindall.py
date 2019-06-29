from scipy.stats import kendalltau
from eval.data import load_system_score
from seaborn import heatmap

if __name__ == '__main__':
    df = load_system_score(normalize_name=True, remove_random_model=True, remove_serban_ppl=True)
    for ds, df2 in df.groupby('dataset'):
        pass

