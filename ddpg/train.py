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

ENV_NAME = 'MountainCarContinuous-v0'
#ENV_NAME = "Pendulum-v1" 

MAX_LOG_SIZE = 1_000_000
BATCH_SIZE = 64
INPUT_SIZE = 3 if ENV_NAME == "Pendulum-v1" else 2
GAMMA = 0.99
NUM_EPOCHS = 1_000
ACTOR_FILEPATH = "actor.p"
CRITIC_FILEPATH = "critic.p"
DATA_FILEPATH = "data.p"
MAX_STEPS = 200 if ENV_NAME == "Pendulum-v1" else 999
MAX_ACTION = 2.0 if ENV_NAME == "Pendulum-v1" else 1.0
MIN_ACTION = -2.0 if ENV_NAME == "Pendulum-v1" else -1.0
MAX_DIST = 2 if ENV_NAME == "Pendulum-v1" else 1
SOFT_UPDATE_RATE = 0.01
PRINT_FREQUENCY = 5
START_NOISE_WEIGHT = 0.00

RELOAD_MODEL = False
RELOAD_DATA = False
USE_SOFT_UPDATE = True
SAVE_DATA = True
USE_PRIORITY = True

actor = load_model(ACTOR_FILEPATH) if RELOAD_MODEL else ActorNetwork(input_size=INPUT_SIZE, max_dist=MAX_DIST)
critic = load_model(CRITIC_FILEPATH) if RELOAD_MODEL else CriticNetwork(input_size=INPUT_SIZE)

env = gym.make(ENV_NAME)

save_model(critic, CRITIC_FILEPATH)
save_model(actor, ACTOR_FILEPATH)
target_actor = load_model(ACTOR_FILEPATH)
target_critic = load_model(CRITIC_FILEPATH)

dataset = load_dataset(DATA_FILEPATH, MAX_LOG_SIZE, USE_PRIORITY) if RELOAD_DATA else (PriorityLog(max_size=MAX_LOG_SIZE) if USE_PRIORITY else SimpleLog(max_size=MAX_LOG_SIZE))

actor_optimizer = optim.AdamW(actor.parameters(), lr=10e-6)
critic_optimizer = optim.AdamW(critic.parameters(), lr=10e-5)
loss_func = nn.MSELoss(reduction='none')

stats_collector = StatsCollector()

for epoch in range(NUM_EPOCHS):

	observation = env.reset()
	total_reward = 0
	done = False
	steps = 0

	noise_to_use = max(START_NOISE_WEIGHT - (START_NOISE_WEIGHT * epoch * 2 / NUM_EPOCHS), 0)
	actor.start_episode()
	actor.set_noise_weight(noise_to_use)

	while not done:

		with torch.no_grad():
			actor.eval()
			action = actor(observation).item()
			actor.train()

		old_observation = observation
		normalized_action = max(min(action, MAX_ACTION), MIN_ACTION)
		observation, reward, done, info = env.step(np.asarray([normalized_action]))
		steps += 1

		dataset.add(old_observation, observation, normalized_action, reward, done)

		if (done and steps < MAX_STEPS):
			print("success!", steps, reward)

		if len(dataset) < (BATCH_SIZE * 3):
			continue

		current_observation_batch, next_observation_batch, action_batch, rewards_batch, incompletions = dataset.sample_batch(BATCH_SIZE)

		with torch.no_grad():
			next_actions_batch = target_actor(next_observation_batch, include_noise=False)
			evaluations = target_critic(next_observation_batch, next_actions_batch)
			targets = rewards_batch.float() + (GAMMA * evaluations.squeeze() * incompletions.squeeze())

		outputs = critic(current_observation_batch, action_batch).squeeze()
		raw_loss = loss_func(outputs, targets)
		loss = torch.mean(raw_loss)
		loss.backward()
		critic_optimizer.step()
		critic_optimizer.zero_grad()

		current_actions = actor(current_observation_batch, include_noise=False)
		raw_value = -1 * critic(current_observation_batch, current_actions).squeeze()
		value = torch.mean(raw_value)
		value.backward()
		actor_optimizer.step()
		actor_optimizer.zero_grad()

		stats_collector.add({"Reward": reward, "Critic Loss": loss.item(), "Actor Loss": value.item(), "Noise": actor.get_noise_weight()})

		if USE_SOFT_UPDATE:
			soft_update(critic, target_critic, SOFT_UPDATE_RATE)
			soft_update(actor, target_actor, SOFT_UPDATE_RATE)

		dataset.record_results(raw_loss)

	if (epoch % PRINT_FREQUENCY == 0) or epoch == NUM_EPOCHS - 1:
		stats_collector.show()
		stats_collector.reset()

	save_model(critic, CRITIC_FILEPATH)
	save_model(actor, ACTOR_FILEPATH)
	save_model(target_actor, "target-actor.p")
	save_model(target_critic, "target-critic.p")

	if not USE_SOFT_UPDATE:
		target_actor = load_model(ACTOR_FILEPATH)
		target_critic = load_model(CRITIC_FILEPATH)

	if SAVE_DATA:
		save_dataset(DATA_FILEPATH, dataset)



