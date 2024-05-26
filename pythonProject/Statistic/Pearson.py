import numpy as np
from scipy.stats import pearsonr


x = np.array([15, 18, 21, 24, 27])
y = np.array([25, 25, 27, 31, 32])


correlation, _ = pearsonr(x, y)
print(correlation)