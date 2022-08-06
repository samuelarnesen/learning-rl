import sys, pickle, time

sys.path.append("/Users/samuelarnesen/Desktop/projects/rl-papers/gym/")
sys.path.append("/usr/local/lib/python3.8/site-packages")

import gym
import torch
import torch.nn as nn
import torch.optim as optim

from models import *


RENDER = True
INCLUDE_NOISE = False
ENV_NAME = "MountainCarContinuous-v0"
#ENV_NAME = "Pendulum-v1"

env = gym.make(ENV_NAME)

FILE_NAME = "actor.p"
CRITIC_FILE = "critic.p"
NUM_TESTS = 100 if not RENDER else 10

model = load_model(FILE_NAME)

critic = load_model(CRITIC_FILE)
target_model = load_model("target-actor.p")
target_critic = load_model("target-critic.p")
model.eval()
critic.eval()
target_model.eval()
target_critic.eval()

total_reward = 0

max_action = -5
min_action = 5
total_action = 0
for i in range(NUM_TESTS):

	total_reward = 0
	done = False

	model.start_episode()
	model.set_noise_weight(0)

	observation = env.reset()
	done = False
	step_count = 0
	critic_estimates = 0
	local_reward = 0

	while not done:

		if RENDER:
			env.render()

		with torch.no_grad():
			action = model(observation=observation, include_noise=INCLUDE_NOISE).item()
			score = critic(observation, torch.tensor(action).unsqueeze(0).unsqueeze(0))
			target_action = target_model(observation=observation, include_noise=INCLUDE_NOISE).item()
			target_score = target_critic(observation, torch.tensor(action).unsqueeze(0).unsqueeze(0))
			#print(round(action, 2), round(target_action, 2))

		observation, reward, done, info = env.step(np.asarray([action]))
		total_reward += reward
		critic_estimates += score
		local_reward += reward
		step_count += 1

		max_action = max(max_action, action)
		min_action = min(min_action, action)
		total_action += action

		if RENDER and done:
			env.render()

	#print("Max: ", round(max_action, 2), "Min: ", round(min_action, 2), "Avg: ", round(total_action / step_count, 2), "Critic", critic_estimates.item(), "Reward", local_reward, "Step Count", step_count)
	outcome = "Success" if step_count < 999 else "\tFailure"
	print(i + 1, "\t", outcome, step_count)
	max_action = -100
	min_action = 100
	total_action = 0


print("Average result: " + str(total_reward / NUM_TESTS))

