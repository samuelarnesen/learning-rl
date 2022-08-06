import sys

sys.path.append("/usr/local/lib/python3.8/site-packages")

import matplotlib.pyplot as plt


rewards = []
with open("results") as f:
	for line in f.readlines():
		if "Reward: " in line:
			split_line = line.split()
			rewards.append(float(split_line[1].strip(",")))

plt.scatter(range(len(rewards)), rewards)
plt.show()
