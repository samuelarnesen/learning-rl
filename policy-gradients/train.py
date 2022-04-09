import sys, pickle, time, random

sys.path.append("/Users/samuelarnesen/Desktop/projects/rl-papers/gym/")
sys.path.append("/usr/local/lib/python3.8/site-packages")

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models import *

env = gym.make('CartPole-v1')

BATCH_SIZE = 8
DISCOUNT_RATE = 0.9
NUM_BATCHES = 10_000_000
PRINT_FREQUENCY = 10_000
AUGMENT_ITERATIONS = 100
SAVE_PATH = "./cartpole-pg.p"

model = PolicyNetwork(num_actions=2)
optimizer = optim.AdamW(model.main_parameters(), lr=1e-5)
value_optimizer = optim.AdamW(model.value_parameters(), lr=1e-4)
augment_optimizer = optim.AdamW(model.main_parameters(), lr=1e-6)
loss_func = nn.MSELoss()

count = 0
avg_reward = 0

for batch_idx in range(NUM_BATCHES):

	batch_discounted_rewards = []
	batch_observations = []
	batch_actions = []
	batch_undiscounted_rewards = []

	for rollout in range(BATCH_SIZE):

		rewards = []
		observations = []
		actions = []
		values = []

		# ======================== rollout ======================== #

		done = False
		observation = env.reset()
		total_reward = 0
		observations.append(observation)
		rewards.append(0.0)

		while not done:

			with torch.no_grad():
			 	action, policy, value = model.act(observation=observation)

			observation, reward, done, info = env.step(action)

			rewards.append(reward)
			actions.append(action)
			values.append(value)
			observations.append(observation)
			total_reward += reward

		idx = random.randrange(0, len(actions))
		discounted_reward = 0
		effective_discount = 1
		for step_reward in rewards[idx + 1:]:
			discounted_reward += (step_reward * effective_discount)
			effective_discount *= DISCOUNT_RATE

		batch_discounted_rewards.append(discounted_reward)
		batch_observations.append(observations[idx])
		batch_actions.append(actions[idx])
		batch_undiscounted_rewards.append(total_reward)

	# ========================  model update ======================== #

	observation_tensor = torch.stack([torch.tensor(ob) for ob in batch_observations])
	rewards_tensor = torch.stack([torch.tensor(reward) for reward in batch_discounted_rewards])

	policies, values = model(observation=observation_tensor)
	action_probs = torch.stack([policies[i, batch_actions[i]] for i in range(BATCH_SIZE)])
	output = torch.mean(action_probs * (rewards_tensor - values) * -1)

	output.backward()
	optimizer.step()
	optimizer.zero_grad()

	with torch.no_grad():
		policies, intermediate = model.calculate_policy(observation_tensor)

	values = model.calculate_value(policies, intermediate)
	value_loss = loss_func(values, rewards_tensor)
	
	value_loss.backward()
	value_optimizer.step()
	value_optimizer.zero_grad()

	new_policies,_ = model.calculate_policy(observation_tensor)
	reversed_policies, _ = model(observation=(observation_tensor * -1).detach())
	augment_loss = loss_func(reversed_policies, (1 - new_policies).clone().detach())

	augment_loss.backward()
	augment_optimizer.step()
	augment_optimizer.zero_grad()

	avg_reward += (sum(batch_undiscounted_rewards) / len(batch_undiscounted_rewards))
	count += 1

	if batch_idx % PRINT_FREQUENCY == 0 or batch_idx == NUM_BATCHES - 1:
		print(round(avg_reward / count, 1))
		count = 0
		avg_reward = 0

		if batch_idx > 0:
			save_model(model, SAVE_PATH)


