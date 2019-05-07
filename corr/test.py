import matplotlib.pyplot as plt

if __name__ == '__main__':
    import seaborn as sns

    sns.set(color_codes=True)
    from corr.utils import UtterScoreDist

    d = UtterScoreDist.from_json_file('save/hred-opensub-distinct_2.json')

    N = 3
    for i in range(N):
        fig = plt.figure()
        ax = plt.subplot()
        sns.distplot(d.utterance, ax=ax)
        fig.savefig('./test.png')
