import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

font_family = plt.rcParams['font.family']

print(font_family)

plt.plot(np.arange(10), np.arange(10))
plt.title('title')
plt.xlabel('bleu_1')
plt.ylabel('adem')
plt.show()
