import sys, pickle, time

sys.path.append("/Users/samuelarnesen/Desktop/projects/rl-papers/gym/")
sys.path.append("/usr/local/lib/python3.8/site-packages")

import gym
import torch
import torch.nn as nn
import torch.optim as optim

from models import *

env = gym.make('Pendulum-v1')

RENDER = True
FILE_NAME = "actor.p"
NUM_TESTS = 10 if not RENDER else 1

model = load_model(FILE_NAME)

total_reward = 0

max_action = -100
min_action = 100
total_action = 0
for i in range(NUM_TESTS):


	observation = env.reset()
	done = False
	step_count = 0

	while not done:

		if RENDER:
			env.render()

		action = model(observation=observation).item()
		observation, reward, done, info = env.step(np.asarray([action]))
		total_reward += reward
		step_count += 1

		max_action = max(max_action, action)
		min_action = min(min_action, action)
		total_action += action

	print("Max: ", round(max_action, 2), "Min: ", round(min_action, 2), "Avg: ", round(total_action / 200, 2))
	#print(total_action / 200)
	max_action = -100
	min_action = 100
	total_action = 0


print("Average result: " + str(total_reward / NUM_TESTS))

