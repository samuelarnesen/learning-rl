import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.append("../utils/")
from base_models import *


class ActorNetwork(nn.Module):

	def __init__(self, input_size=8):
		super().__init__()
		self.__core = MLP(intermediates=[16, 16], input_size=input_size, output_size=1)
		self.__noisy_channel = NoisyChannel(0.2)

	def forward(self, observation, include_noise=True, batch_size=1, max_dist=2):
		observation_tensor = observation if torch.is_tensor(observation) else torch.tensor(observation).squeeze().unsqueeze(0)
		noise = self.__noisy_channel(batch_size) if include_noise and self.get_noise_weight() > 0 else torch.zeros(batch_size)
		return max_dist * torch.tanh(self.__core(observation_tensor) + noise.unsqueeze(1))

	def parameters(self):
		return self.__core.parameters() + [param for param in self.__noisy_channel.parameters()]

	def get_noise_weight(self):
		return self.__noisy_channel.get_weight()

class CriticNetwork(nn.Module):

	def __init__(self, input_size=8):
		super().__init__()
		self.__state_encoder = MLP(intermediates=[16], input_size=input_size, output_size=8)
		self.__action_encoder = MLP(intermediates=[16], input_size=1, output_size=4)
		self.__final_layer = MLP(intermediates=[16], input_size=12, output_size=1)

	def forward(self, observation, action):
		observation_tensor = observation if torch.is_tensor(observation) else torch.tensor(observation).squeeze().unsqueeze(0)
		action_tensor = action if torch.is_tensor(action) else torch.tensor(action).squeeze().unsqueeze(0)
		state_encoding = self.__state_encoder(observation_tensor)
		action_encoding = self.__action_encoder(action_tensor)
		state_action_encoding = torch.cat([state_encoding, action_encoding], dim=1)
		return self.__final_layer(F.relu(state_action_encoding))

	def parameters(self):
		return self.__state_encoder.parameters() + self.__action_encoder.parameters() + self.__final_layer.parameters()


class ActorCriticModel(nn.Module):

	def __init__(self, input_size=8, state_embedding_size=4, action_embedding_size=2):
		super().__init__()
		self.__state_encoder = MLP(intermediates=[4], input_size=input_size, output_size=state_embedding_size)
		self.__action_encoder = MLP(intermediates=[2], input_size=1, output_size=action_embedding_size)
		self.__final_critic_layer = MLP(intermediates=[4], input_size=(state_embedding_size + action_embedding_size), output_size=1)
		self.__final_actor_layer = MLP(intermediates=[2], input_size=state_embedding_size, output_size=1)

	def act(self, observation):
		observation_tensor = observation if torch.is_tensor(observation) else torch.tensor(observation).squeeze().unsqueeze(0)
		state = self.__state_encoder(observation_tensor)
		return self.__final_actor_layer(F.relu(state))

	def critique(self, observation, action):
		observation_tensor = observation if torch.is_tensor(observation) else torch.tensor(observation).squeeze().unsqueeze(0)
		action_tensor = action if torch.is_tensor(action) else torch.tensor(action).squeeze().unsqueeze(0)
		state_encoding = self.__state_encoder(observation_tensor)
		action_encoding = self.__action_encoder(action_tensor)
		state_action_encoding = torch.cat([state_encoding, action_encoding], dim=1)
		return self.__final_critic_layer(F.relu(state_action_encoding))

	def actor_parameters(self):
		return self.__final_actor_layer.parameters()

	def critic_parameters(self):
		return self.__state_encoder.parameters() + self.__action_encoder.parameters() + self.__final_critic_layer.parameters()



