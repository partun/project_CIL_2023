import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


POS_FULL = '../twitter-datasets/train_pos_full_swht.txt'
NEG_FULL = '../twitter-datasets/train_neg_full_swht.txt'
FILEPATHS = [POS_FULL, NEG_FULL]


lengths = []
for filename in FILEPATHS:
    with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                lengths.append(len(line.rstrip().split(" "))) 

sns.set()
plt.rcParams['figure.figsize'] = (16, 9)
f = plt.figure()
f.add_subplot()
sns.distplot(lengths)

plt.ylabel("density", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title("tweets length counts", fontsize=20)
plt.show()