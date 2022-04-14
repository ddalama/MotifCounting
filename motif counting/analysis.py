import re
from scipy import stats
from scipy.io import loadmat
import numpy as np

result = []

with open('motifcounts.txt', 'r') as f:

    lines = f.readlines()

    for line in lines:
        nums = re.findall(r': (\d+)', line)
        if len(nums) > 2:
            print(nums)
            nums = np.array([int(n) for n in nums])
            result.append(nums)

result = np.array(result)

print(len(result))


data = loadmat("BP.mat")

labels = data['label']
labels = np.array([l[0] for l in labels])

g1 = result[labels == -1]
g2 = result[labels == 1][:g1.shape[0], :]

name = ['m4.1', 'm4.3', 'm4.2', 'm4.5', 'm4.6', 'm4.4']

for i in range(result.shape[1]):
    s, p = stats.ttest_ind(g1[:, i], g2[:, i])
    print(f"|{name[i]}|{np.mean(g1[:, i]):.2f}+={np.std(g1[:, i]):.2f}|{np.mean(g2[:, i]):.2f}+={np.std(g2[:, i]):.2f}|{p:.3f}|")
