import sys, pickle, time

sys.path.append("/Users/samuelarnesen/Desktop/projects/rl-papers/gym/")
sys.path.append("/usr/local/lib/python3.8/site-packages")

import gym
import torch
import torch.nn as nn
import torch.optim as optim

from models import *
from dataset import *

env = gym.make('CartPole-v1')

FILE_NAME = "cartpole-dqn.p"
NUM_TESTS = 1
RENDER = True
PRINT_INTERMEDIATE = True

with open(FILE_NAME, "rb") as f:
	model = pickle.load(f)

total_reward = 0

for i in range(NUM_TESTS):

	observation = env.reset()
	done = False
	step_count = 0

	while not done:

		if RENDER:
			env.render()

		action, _ = model.act(observation=observation, greedy=False, epsilon=0)
		observation, reward, done, info = env.step(action)
		total_reward += reward
		step_count += 1

	if PRINT_INTERMEDIATE:
		print("\t", step_count)


print("Average result: " + str(total_reward / NUM_TESTS))

