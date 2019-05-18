from pathlib import Path

from eval.data import load_system_score

import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = load_system_score(
        prefix=Path('/home/cgsdfc/Metrics/Eval/data/v2/score/db'),
        remove_random_model=True,
    )
    bleu_2_all = df[df.metric == 'bleu_2']
    plt.plot()
