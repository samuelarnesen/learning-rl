import sys, pickle, time, random, os, copy

sys.path.append("/Users/samuelarnesen/Desktop/projects/rl-papers/gym/")
sys.path.append("/usr/local/lib/python3.8/site-packages")
sys.path.append("/Users/samuelarnesen/Desktop/projects/rl-papers/utils/")

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models import *
from base_datasets import *
from stats_collector import StatsCollector

MAX_LOG_SIZE = 10_000_000
NUM_ROLLOUTS_PER_EPOCH = 300
NUM_UPDATES_PER_EPOCH = 2000
BATCH_SIZE = 128
INPUT_SIZE = 3
GAMMA = 0.99
NUM_EPOCHS = 1_000_000
ACTOR_FILEPATH = "actor.p"
CRITIC_FILEPATH = "critic.p"
MAX_STEPS = 200
MAX_ACTION = 2.0
MIN_ACTION = -2.0
SOFT_UPDATE_RATE = 0.05

RELOAD = False

env = gym.make('Pendulum-v1')

actor = load_model(ACTOR_FILEPATH) if RELOAD else ActorNetwork(input_size=INPUT_SIZE)
critic = load_model(CRITIC_FILEPATH) if RELOAD else CriticNetwork(input_size=INPUT_SIZE)

save_model(critic, CRITIC_FILEPATH)
save_model(actor, ACTOR_FILEPATH)
target_actor = load_model(ACTOR_FILEPATH)
target_critic = load_model(CRITIC_FILEPATH)

dataset = SimpleLog(max_size=MAX_LOG_SIZE)

actor_optimizer = optim.AdamW(actor.parameters(), lr=10e-6)
critic_optimizer = optim.AdamW(critic.parameters(), lr=10e-4)
loss_func = nn.MSELoss()

update_stats_collector = StatsCollector()
rollout_stats_collector = StatsCollector()

def flip_observation(ob):
	new_ob = copy.deepcopy(ob)
	new_ob[1] = ob[1]
	return new_ob

# ======================= ROLL-OUT ========================== #

for epoch in range(NUM_EPOCHS):

	for rollout in range(NUM_ROLLOUTS_PER_EPOCH):

		observation = env.reset()
		total_reward = 0
		done = False
		steps = 0

		while not done:

			with torch.no_grad():
				action = actor(observation).item()

			old_observation = observation
			normalized_action = max(min(action, MAX_ACTION), MIN_ACTION)
			observation, reward, done, info = env.step(np.asarray([normalized_action]))

			dataset.add(old_observation, observation, normalized_action, reward)
			#dataset.add(flip_observation(old_observation), flip_observation(observation), normalized_action * -1, reward) # augments data

			total_reward += reward

		rollout_stats_collector.add({"Reward": total_reward})

	rollout_stats_collector.show()
	rollout_stats_collector.reset()
		
	for i in range(NUM_UPDATES_PER_EPOCH):

		current_observation_batch, next_observation_batch, action_batch, rewards_batch = dataset.sample_batch(BATCH_SIZE)

		with torch.no_grad():
			next_actions_batch = target_actor(next_observation_batch, include_noise=False)
			evaluations = target_critic(next_observation_batch, next_actions_batch)
			targets = rewards_batch.float() + (GAMMA * evaluations.squeeze())

		outputs = critic(current_observation_batch, action_batch).squeeze()
		loss = loss_func(outputs, targets)
		loss.backward()
		critic_optimizer.step()
		critic_optimizer.zero_grad()

		current_actions = actor(current_observation_batch)
		raw_value = -1 * (rewards_batch + critic(current_observation_batch, current_actions).squeeze())
		value = torch.mean(raw_value)
		value.backward()
		actor_optimizer.step()
		actor_optimizer.zero_grad()

		update_stats_collector.add({"Critic Loss": loss.item(), "Actor Loss": value.item()})

		soft_update(critic, target_critic, SOFT_UPDATE_RATE)
		soft_update(actor, target_actor, SOFT_UPDATE_RATE)


	update_stats_collector.show()
	update_stats_collector.reset()

	save_model(critic, CRITIC_FILEPATH)
	save_model(actor, ACTOR_FILEPATH)
	print("Noise:", actor.get_noise_weight())




















