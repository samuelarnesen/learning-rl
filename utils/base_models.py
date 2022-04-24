import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, random, pickle

class NoisyChannel(nn.Module):

	def __init__(self, base_weight=0.1):
		super().__init__()
		self.__weight = nn.Parameter(torch.rand(1) * base_weight)

	def forward(self, size):
		return torch.randn(size) * self.__weight

	def get_weight(self):
		return self.__weight.item()

class MLP(nn.Module):

	def __init__(self, intermediates, input_size=6, output_size=1, activation_func=F.relu):
		super().__init__()
		dims = [input_size] + intermediates + [output_size]
		self.__layers = [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
		self.__activation_func = activation_func

	def forward(self, input_batch):
		current_batch = input_batch
		for layer in self.__layers[:-1]:
			current_batch = self.__activation_func(layer(current_batch))
		return self.__layers[-1](current_batch)

	def parameters(self):
		parameters = []
		for layer in self.__layers:
			for parameter in layer.parameters():
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

def soft_update(reference, target, rate):
	for ref_param, target_param in zip(reference.parameters(), target.parameters()):
		target_param = (rate * ref_param) + ((1 - rate) * target_param)