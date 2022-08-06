import sys, pickle, time

sys.path.append("/Users/samuelarnesen/Desktop/projects/rl-papers/gym/")
sys.path.append("/usr/local/lib/python3.8/site-packages")
sys.path.append("../utils/")

import gym
import torch
import torch.nn as nn
import torch.optim as optim

from models import *
from base_models import *

RENDER = True
ENV_NAME = "CartPole-v1"

env = gym.make(ENV_NAME)

FILE_NAME = "model.p"
CRITIC_FILE = "critic.p"
NUM_TESTS = 100 if not RENDER else 25

model = load_model(FILE_NAME)
model.eval()

total_reward = 0
run_reward = 0

for i in range(NUM_TESTS):

	done = False
	observation = env.reset()
	done = False
	step_count = 0
	critic_estimates = 0
	run_reward = 0

	while not done:

		if RENDER:
			env.render()

		with torch.no_grad():
			action = model.act(observation=observation)

		observation, reward, done, info = env.step(action.item())
		#print("\t", action.item(), torch.mean(model.evaluate(observation)).item(), "\n\n\n")
		run_reward += reward

		if RENDER and done:
			env.render()

	print(run_reward)
	total_reward += run_reward

print("Average result: " + str(total_reward / NUM_TESTS))
