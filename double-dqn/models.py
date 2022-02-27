import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, random, pickle

class MLP(nn.Module):

	def __init__(self, intermediates, input_size=6, output_size=1):

		super().__init__()
		dims = [input_size] + intermediates + [output_size]
		self.__layers = [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]

	def forward(self, input_batch):
		current_batch = input_batch
		for layer in self.__layers[:-1]:
			current_batch = F.relu(layer(current_batch))
		return self.__layers[-1](current_batch)

	def parameters(self):
		parameters = []
		for layer in self.__layers:
			for parameter in layer.parameters():
				parameters.append(parameter)
		return parameters

class DuelingQNetwork(nn.Module):

	def __init__(self, legal_actions):
		super().__init__()
		self.__trunk = MLP(intermediates=[4, 4], input_size=4, output_size=8)
		self.__value_network = nn.Linear(4, 1)
		self.__advantage_network = nn.Linear(4, len(legal_actions))
		self.__legal_actions = legal_actions

	def act(self, observation, greedy=True, epsilon=0.1):
		state = torch.tensor(observation).unsqueeze(0)
		_, advantages = self(state)
		action_idx = torch.argmax(advantages.squeeze()).item()
		if not greedy and random.random() < epsilon:
			action_idx = random.randrange(len(self.__legal_actions))
		return action_idx, advantages.squeeze()

	def batch_act(self, observation):
		_, advantages = self(observation)
		action_idxs = torch.argmax(advantages, dim=1)
		return action_idxs

	def evaluate(self, observations_tensor, actions):
		value, advantages = self(observations_tensor)
		max_advantages = torch.max(advantages, dim=1).values.unsqueeze(1).expand(-1, len(self.__legal_actions))
		adjusted_advantages = value + (advantages - max_advantages)
		return torch.gather(input=adjusted_advantages, dim=1, index=actions).squeeze()

	def forward(self, state):
		intermediate = self.__trunk(state)
		split_output = torch.split(intermediate, 4, dim=1)
		value = self.__value_network(F.relu(split_output[0]))
		advantages = self.__advantage_network(F.relu(split_output[1]))
		return value, advantages

	def parameters(self):
		parameters = []
		for parameter in self.__trunk.parameters():
			parameters.append(parameter)
		for parameter in self.__value_network.parameters():
			parameters.append(parameter)
		for parameter in self.__advantage_network.parameters():
			parameters.append(parameter)	
		return parameters


class SimpleQNetwork(nn.Module):

	def __init__(self, legal_actions):
		super().__init__()
		self.__scorer = MLP(intermediates=[4, 4, 2], input_size=4, output_size=len(legal_actions))
		self.__legal_actions = legal_actions

	def act(self, observation, greedy=True, epsilon=0.1):
		state = torch.tensor(observation).unsqueeze(0)
		scores = self(state).squeeze()
		action_idx = torch.argmax(scores).item()
		if not greedy and random.random() < epsilon:
			action_idx = random.randrange(len(self.__legal_actions))
		return action_idx, scores

	def batch_act(self, observation):
		scores = self(observation)
		action_idxs = torch.argmax(scores, dim=1)
		return action_idxs

	def forward(self, state):
		return self.__scorer(state)

	def evaluate(self, observations_tensor, actions):
		scores = self(observations_tensor)
		return torch.gather(input=scores, dim=1, index=actions).squeeze()

	def parameters(self):
		parameters = []
		for parameter in self.__scorer.parameters():
			parameters.append(parameter)
		return parameters

def save_model(model, file_name):
	with open(file_name, "wb") as f:
		pickle.dump(model, f)

def load_model(file_name):
	with open(file_name, "rb") as f:
		return pickle.load(f)

def deep_copy(model, file_name):
	save_model(model, file_name)
	return load_model(file_name)
