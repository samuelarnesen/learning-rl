import random, math, sys
import torch

class LogEntry:

	def __init__(self, current_observation, next_observation, action, reward):
		self.current_observation = current_observation
		self.next_observation = next_observation
		self.action = action
		self.reward = reward

class SimpleLog:

	def __init__(self, max_size=None, auto_trim=True):
		self.__log = []
		self.__max_size = max_size
		self.__auto_trim = auto_trim and max_size != None

	def add(self, current_observation, next_observation, action, reward):
		self.__log.append(LogEntry(current_observation, next_observation, action, reward))
		if self.__auto_trim:
			self.trim()

	def sample(self):
		return random.choice(self.__log)

	def sample_batch(self, batch_size):
		log_entries = random.choices(self.__log, k=batch_size)
		return self.construct_batch_from_sample(log_entries)

	def construct_batch_from_sample(self, log_entries):
		observations = torch.stack([torch.tensor(entry.current_observation).squeeze() for entry in log_entries]).float()
		next_observations = torch.stack([torch.tensor(entry.next_observation).squeeze() for entry in log_entries]).float()

		actions = torch.tensor([entry.action for entry in log_entries]).unsqueeze(1)
		rewards = torch.tensor([entry.reward for entry in log_entries])
		return observations, next_observations, actions, rewards

	def trim(self):
		if self.__max_size != None and len(self.__log) > self.__max_size:
			self.__log = self.__log[int((len(self.__log) / 2)):]

	def get_max_size(self):
		return self.__max_size

	def get_log(self):
		return self.__log

	def reset(self):
		self.__log = []

	def __len__(self):
		return len(self.__log)