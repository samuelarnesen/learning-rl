import sys, math, random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DistributionalModel(nn.Module):

	def __init__(self, input_size=6, quantiles=51, num_actions=3):
		super().__init__()

		HL_1 = 16
		HL_2 = 32

		self.__layer_one = nn.Linear(input_size, HL_1)
		self.__normalizer_one = nn.BatchNorm1d(HL_1)
		self.__layer_two = nn.Linear(HL_1, HL_2)
		self.__normalizer_two = nn.BatchNorm1d(HL_2)
		self.__final_layer = nn.Linear(HL_2, num_actions * quantiles)

		self.__quantiles = quantiles
		self.__num_actions = num_actions

	def forward(self, observation):
		observation_tensor = observation if torch.is_tensor(observation) else torch.tensor(observation).squeeze().unsqueeze(0)
		state_encoding = self.__normalizer_one(F.gelu(self.__layer_one(observation_tensor)))
		penultimate = self.__normalizer_two(F.gelu(self.__layer_two(state_encoding)))
		output = self.__final_layer(penultimate)
		return output

	def act(self, observation, greedy=True, epsilon=0.05):
		output = self(observation)
		batch_size, _ = output.size()
		reshaped_output = torch.reshape(output, [batch_size, self.__num_actions, self.__quantiles])
		expectations = torch.mean(reshaped_output, dim=2)
		
		if not greedy and random.random() < epsilon and batch_size == 1:
			return torch.tensor(random.randrange(self.__num_actions))

		return torch.argmax(expectations, dim=1)

	def evaluate(self, observation, actions=None):
		output = self(observation)
		batch_size, _ = output.size()
		reshaped_output = torch.reshape(output, [batch_size, self.__num_actions, self.__quantiles]) # [batch_size, num_actions, num_quantiles]
		expectations = torch.mean(reshaped_output, dim=2)
		if actions == None:
			actions = torch.argmax(expectations, dim=1)
		distributions = torch.stack([reshaped_output[i, actions[i], :] for i in range(batch_size)])
		return distributions

def quantile_huber_loss(prediction, target, kappa=10):
	batch_size, num_quantiles = prediction.size()
	zero_tensor = torch.zeros(batch_size, num_quantiles)
	quantile_tensor = torch.tensor(np.asarray([(i + 1) / (num_quantiles + 1) for i in range(num_quantiles)])).unsqueeze(0).repeat(batch_size, 1)
	diff = prediction - target
	underestimates = torch.minimum(diff, zero_tensor)**2
	overestimates = torch.maximum(diff, zero_tensor)**2

	raw_loss = (quantile_tensor * underestimates) + ((1 - quantile_tensor) * overestimates)
	return torch.mean(raw_loss)






