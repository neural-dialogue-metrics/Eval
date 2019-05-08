import matplotlib

matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


def corrfunc(x, y, **kwargs):
    (r, p) = pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f} ".format(r),
                xy=(.1, 0.9), xycoords=ax.transAxes)
    ax.annotate("p = {:.3f}".format(p),
                xy=(.1, 0.85), xycoords=ax.transAxes)


df = sns.load_dataset("iris")
df = df[df["species"] == "setosa"]
graph = sns.pairplot(df)
graph.map(corrfunc)
plt.show()
