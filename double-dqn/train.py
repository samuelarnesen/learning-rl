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

NUM_EPISODES_PER_CYCLE = 500
NUM_CYCLES = 400
MAX_STEPS = 500
TARGET_UPDATE_FREQUENCY = 500
EPSILON = 0.2
MINIBATCH_SIZE = 64
NON_GREEDY_FREQUENCY = 100
NUM_MINIBATCHES = 2_000
REWARD_PER_FRAME = 1
MAX_DATASET_SIZE = 3_000_000
DEATH_PENALTY = -10
USE_PRIORITY = False
GAMMA = 0.90
USE_DOUBLE = True
USE_DUEL = True
MODEL_PATH = "./cartpole-dqn.p"

legal_actions = [0, 1]
model = DuelingQNetwork(legal_actions=legal_actions) if USE_DUEL else SimpleQNetwork(legal_actions=legal_actions)
target_model = None

dataset = PrioritizedLog(MAX_DATASET_SIZE) if USE_PRIORITY else SimpleLog(MAX_DATASET_SIZE)
reduction = 'none' if USE_PRIORITY else 'mean'

loss_func = nn.MSELoss(reduction=reduction)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

for cycle in range(NUM_CYCLES):

    epsilon = EPSILON * (1 - (cycle / NUM_CYCLES / 2))

    # ========================  online data gathering ======================== #

    for episode in range(NUM_EPISODES_PER_CYCLE): 

        observation = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        previous_observation = None
        greedy = episode != NUM_EPISODES_PER_CYCLE - 1

        while step_count < MAX_STEPS and not done:

            action, _ = model.act(observation=observation, greedy=greedy, epsilon=epsilon)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1

            if not isinstance(previous_observation, type(None)):
                reward_to_use = reward if not done else DEATH_PENALTY
                dataset.add_with_augmentation(previous_observation, observation, action, reward_to_use)

            previous_observation = observation

        if not greedy:
            print(int(total_reward))

    dataset.trim()

    # ========================  offline training ======================== #

    live_rate = 0
    num_samples = 0
    for i in range(NUM_MINIBATCHES):

        if i % TARGET_UPDATE_FREQUENCY == 0:
            target_model = deep_copy(model, MODEL_PATH)

        observations, next_observations, actions, rewards = dataset.sample_batch(MINIBATCH_SIZE)
        live_positions = (rewards > DEATH_PENALTY)
        live_rate += torch.sum(live_positions).item()
        num_samples += MINIBATCH_SIZE

        with torch.no_grad():
            if USE_DOUBLE:
                model_actions = model.batch_act(observation=next_observations).unsqfueeze(1)
                target_scores = target_model.evaluate(next_observations, model_actions)
                targets = (rewards + (GAMMA * target_scores)) * live_positions
            else:
                target_scores = target_model(next_observations)
                targets = (rewards + (GAMMA * torch.max(target_scores, dim=1).values)) * live_positions

        with torch.enable_grad():
            estimates = model.evaluate(observations, actions)
            raw_loss = loss_func(estimates, targets)
            dataset.record_results(raw_loss)
            loss = torch.mean(raw_loss) if USE_PRIORITY else raw_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    scheduler.step()
