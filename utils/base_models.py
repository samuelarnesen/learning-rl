import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, random, pickle
import numpy as np

class NoisyChannel:

	def __init__(self, base_weight=0.1, width=2):
		super().__init__()
		self.__weight = torch.ones(1) * base_weight
		self.__starter = self.fit_to_range(torch.randn(1), width)
		self.__width = width

	def get_noise(self, size, alpha=0.8):
		random_value = self.fit_to_range(torch.randn(size), self.__width)
		self.__starter = (alpha * self.__starter) + ((1 - alpha) * random_value)
		return self.__starter * self.__weight

	def get_weight(self):
		return self.__weight.item()

	def set_weight(self, weight):
		self.__weight = torch.tensor(weight)

	def start_episode(self):
		self.__starter = self.fit_to_range(torch.randn(1), self.__width)

	def fit_to_range(self, input_val, width):
		return ((input_val * 2) - 1) * width

class MLP(nn.Module):

	def __init__(self, intermediates, input_size=6, output_size=1, activation_func=F.relu):
		super().__init__()
		dims = [input_size] + intermediates + [output_size]
		self.__layers = [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
		self.__activation_func = activation_func
		self.__batch_norms = [nn.BatchNorm1d(dim) for dim in dims[1:]]

	def forward(self, input_batch, normalize=False):
		current_batch = input_batch
		for i, layer in enumerate(self.__layers[:-1]):
			current_batch = self.__activation_func(layer(current_batch))
			if normalize:
				current_batch = self.__batch_norms[i](current_batch)
		return self.__layers[-1](current_batch)

	def parameters(self):
		parameters = []
		for layer in self.__layers:
			for parameter in layer.parameters():
				parameters.append(parameter)
		for batch_norm in self.__batch_norms:
			for parameter in batch_norm.parameters():
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
		target_param.data.copy_((rate * ref_param) + ((1 - rate) * target_param))