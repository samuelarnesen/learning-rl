import sys, pickle, time

sys.path.append("/Users/samuelarnesen/Desktop/projects/rl-papers/gym/")
sys.path.append("/usr/local/lib/python3.8/site-packages")

import gym
import torch
import torch.nn as nn
import torch.optim as optim

from models import *

INCLUDE_NOISE = False
ENV_NAME = "MountainCarContinuous-v0"
#ENV_NAME = "Pendulum-v1"

env = gym.make(ENV_NAME)

FILE_NAME = "actor.p"
CRITIC_FILE = "critic.p"

model = load_model(FILE_NAME)
critic = load_model(CRITIC_FILE)
model.eval()
critic.eval()

min_action = -2
max_action = 2

step_size = 0.1


observation = env.reset()
done = False
step_count = 0

while not done:

	action = model(observation=observation, include_noise=INCLUDE_NOISE).item()

	observation, reward, done, info = env.step(np.asarray([action]))
	score = critic(observation, torch.tensor(action).unsqueeze(0).unsqueeze(0)).item()

	possible_action = min_action
	while possible_action < max_action:
		possible_score = critic(observation, torch.tensor(float(possible_action)).unsqueeze(0).unsqueeze(0)).item()
		print(step_count, "\t", round(possible_action, 2), round(possible_score, 2), "\t", round(action, 2), round(score, 2))
		possible_action += step_size

	print()




	step_count += 1

print(step_count)


