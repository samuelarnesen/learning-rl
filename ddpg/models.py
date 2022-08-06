import sys, math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.append("../utils/")
from base_models import *


class Noise:

	def __init__(self, mu, sigma, theta, step, weight=0.005, initial=None):
		self.__mu = mu 
		self.__sigma = sigma
		self.__theta = theta
		self.__step = step 
		self.__weight = weight
		self.__initial = initial
		self.__current = initial

	def get_noise(self):
		part_one = (self.__mu - self.__current) * self.__theta * self.__step
		part_two = self.__sigma * math.sqrt(self.__step)
		random_value = np.random.normal(size=1)[0]
		self.__current += part_one + part_two + random_value
		return self.__current * self.__weight

	def get_weight(self):
		return self.__weight 

	def set_weight(self, weight):
		self.__weight = weight

	def reset(self):
		self.__current = self.__initial if self.__initial != None else 0


class ActorNetwork(nn.Module):

	def __init__(self, input_size=8, max_dist=2, hidden_size=64):
		super().__init__()
		self.__layer_one = nn.Linear(input_size, hidden_size)
		self.__norm_one = nn.BatchNorm1d(hidden_size)
		self.__layer_two = nn.Linear(hidden_size, hidden_size)
		self.__norm_two = nn.BatchNorm1d(hidden_size)
		self.__layer_three = nn.Linear(hidden_size, 1)
		self.__noise = Noise(mu=0, sigma=0.2, theta=0.15, step=0.01, initial=None)
		self.__max_dist = max_dist

	def forward(self, observation, include_noise=True, batch_size=1):
		observation_tensor = observation if torch.is_tensor(observation) else torch.tensor(observation).squeeze().unsqueeze(0)
		noise = self.__noise.get_noise() if include_noise and batch_size == 1 else 0
		intermediate_one = self.__norm_one(F.gelu(self.__layer_one(observation_tensor)))
		intermediate_two = self.__norm_two(F.gelu(self.__layer_two(intermediate_one)))
		final = self.__layer_three(intermediate_two)
		return (self.__max_dist * torch.tanh(final)) + noise

	def get_noise_weight(self):
		return self.__noise.get_weight()

	def set_noise_weight(self, weight):
		self.__noise.set_weight(weight)

	def start_episode(self):
		self.__noise.reset()

class CriticNetwork(nn.Module):

	def __init__(self, input_size=8):
		super().__init__()
		self.__state_encoder = nn.Linear(input_size, 15)
		self.__normalizer_one = nn.BatchNorm1d(16)
		self.__layer_two = nn.Linear(16, 32)
		self.__normalizer_two = nn.BatchNorm1d(32)
		self.__final_layer = nn.Linear(32, 1)n

	def forward(self, observation, action):
		observation_tensor = observation if torch.is_tensor(observation) else torch.tensor(observation).squeeze().unsqueeze(0)
		action_tensor = action if torch.is_tensor(action) else torch.tensor(action).squeeze().unsqueeze(0)
		state_encoding = F.gelu(self.__state_encoder(observation_tensor))
		state_action_encoding = self.__normalizer_one((torch.cat([state_encoding, action_tensor], dim=1)))
		penultimate = self.__normalizer_two(F.gelu(self.__layer_two(state_action_encoding)))
		output = self.__final_layer(penultimate)
		return output



