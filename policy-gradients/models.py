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
			current_batch = torch.tanh(layer(current_batch))
		return self.__layers[-1](current_batch)

	def parameters(self):
		parameters = []
		for layer in self.__layers:
			for parameter in layer.parameters():
				parameters.append(parameter)
		return parameters

class PolicyNetwork(nn.Module):

	def __init__(self, num_actions):
		super().__init__()
		self.__core = MLP(intermediates=[4, 4], input_size=4, output_size=4)
		self.__advantage_network = nn.Linear(4, num_actions)
		self.__action_network = nn.Linear(4, num_actions)
		self.__softmax = nn.Softmax(dim=1)

	def calculate_policy(self, observation):
		observation_tensor = observation if torch.is_tensor(observation) else torch.tensor(observation).unsqueeze(0)
		intermediate = self.__core(observation_tensor)
		scores = self.__action_network(torch.tanh(intermediate))

		policy = self.__softmax(scores).squeeze()
		return policy, intermediate

	def calculate_value(self, policy, intermediate):
		advantages = self.__advantage_network(torch.tanh(intermediate))
		#return torch.sum(advantages * policy, dim=1)
		return torch.max(advantages, dim=1).values

	def forward(self, observation):
		policy, intermediate = self.calculate_policy(observation)
		value = self.calculate_value(policy, intermediate)
		return policy, value

	def act(self, observation):
		policy, value = self(observation)
		action = torch.multinomial(policy, 1).item()
		return action, policy, value

	def main_parameters(self):
		parameters = []
		for parameter in self.__core.parameters():
			parameters.append(parameter)
		for parameter in self.__softmax.parameters():
			parameters.append(parameter)
		for parameter in self.__action_network.parameters():
			parameters.append(parameter)	
		return parameters

	def value_parameters(self):
		return self.__advantage_network.parameters()

def save_model(model, file_name):
	with open(file_name, "wb") as f:
		pickle.dump(model, f)

def load_model(file_name):
	with open(file_name, "rb") as f:
		return pickle.load(f)

def deep_copy(model, file_name):
	save_model(model, file_name)
	return load_model(file_name)