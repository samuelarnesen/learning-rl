import sys, pickle, time

sys.path.append("/Users/samuelarnesen/Desktop/projects/rl-papers/gym/")
sys.path.append("/usr/local/lib/python3.8/site-packages")

import gym
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append("../utils/")

from base_models import *
from models import *
from base_datasets import *
from stats_collector import StatsCollector

#env = gym.make('Acrobot-v1')
env = gym.make('CartPole-v1')

MAX_DATASET_SIZE = 10_000_000
BATCH_SIZE = 64
USE_PRIORITY = False
NUM_EPOCHS = 100_000
EPSILON = 0.1
DISCOUNT = 0.99
UNGREEDY_FREQUENCY = 10
NUM_UPDATES_PER_ROLLOUT = 4
SAVE_PATH = "model.p"
PRINT_FREQUENCY = 1000
QUANTILES = 51
TARGET_UPDATE_FREQUENCY = 250
SUCCESSFUL_RUNS = 25

model = DistributionalModel(input_size=4, num_actions=2, quantiles=QUANTILES)
dataset = PrioritizedLog(MAX_DATASET_SIZE) if USE_PRIORITY else SimpleLog(MAX_DATASET_SIZE)
reduction = 'none' if USE_PRIORITY else 'mean'

optimizer = optim.AdamW(model.parameters(), lr=1e-4)
rollout_stats_collector = StatsCollector()
update_stats_collector = StatsCollector()

save_model(model, SAVE_PATH)
target_model = load_model(SAVE_PATH)
previous_success = 0

for epoch in range(NUM_EPOCHS):

	epsilon = EPSILON * (1 - (epoch / NUM_EPOCHS / 2))
	model.eval()

	observation = env.reset()
	done = False
	run_reward = 0
	greedy = (epoch % UNGREEDY_FREQUENCY != 0) or previous_success > 0

	while not done:
		old_observation = observation
		with torch.no_grad():
			action = model.act(observation=observation, greedy=(epoch % UNGREEDY_FREQUENCY != 0), epsilon=epsilon)

		observation, reward, done, info = env.step(action.item())
		run_reward += reward
		if not done or run_reward < 499:
			dataset.add(old_observation, observation, action, reward, done)

	rollout_stats_collector.add({"Reward": run_reward})
	if run_reward == 500:
		print("Success")
		previous_success += 1
		if previous_success > SUCCESSFUL_RUNS:
			print("Finished successfully")
			save_model(model, SAVE_PATH)
			sys.exit()
		continue
	previous_success = 0

	for i in range(max(NUM_UPDATES_PER_ROLLOUT, int(run_reward / BATCH_SIZE))):
		model.train()
		current_observation_batch, next_observation_batch, action_batch, rewards_batch, incompletions = dataset.sample_batch(BATCH_SIZE)

		with torch.no_grad():
			target = (DISCOUNT * target_model.evaluate(next_observation_batch) * incompletions.repeat(1, QUANTILES)) + rewards_batch.unsqueeze(1).repeat(1, QUANTILES)

		optimizer.zero_grad()
		prediction = model.evaluate(current_observation_batch, action_batch.squeeze())
		loss = quantile_huber_loss(prediction, target)
		loss.backward()
		optimizer.step()
		
		update_stats_collector.add({"Loss": loss.item()})

	if epoch % TARGET_UPDATE_FREQUENCY == 0:
		save_model(model, SAVE_PATH)
		target_model = load_model(SAVE_PATH)

	if epoch % PRINT_FREQUENCY == 0 or epoch == NUM_EPOCHS - 1:
		rollout_stats_collector.show()
		rollout_stats_collector.reset()
		update_stats_collector.show()
		update_stats_collector.reset()
		print()











