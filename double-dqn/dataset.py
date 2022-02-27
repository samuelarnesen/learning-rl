import random, math, sys
import torch

class LogEntry:

	def __init__(self, current_observation, next_observation, action, reward):
		self.current_observation = current_observation
		self.next_observation = next_observation
		self.action = action
		self.reward = reward

class SimpleLog:

	def __init__(self, max_size=None):
		self.__log = []
		self.__max_size = max_size

	def add_with_augmentation(self, current_observation, next_observation, action, reward):
		def flip_observation(observation):
			return [observation[0] * -1, observation[1] * -1, observation[2] * -1, observation[3]]
		def flip_action(action):
			return (action - 1) * -1
		self.add(current_observation, next_observation, action, reward)
		self.add(flip_observation(current_observation), flip_observation(next_observation), flip_action(action), reward)

	def add(self, current_observation, next_observation, action, reward):
		self.__log.append(LogEntry(current_observation, next_observation, action, reward))

	def sample_batch(self, batch_size):
		log_entries = random.choices(self.__log, k=batch_size)
		return self.construct_batch_from_sample(log_entries)

	def construct_batch_from_sample(self, log_entries):
		observations = torch.stack([torch.tensor(entry.current_observation) for entry in log_entries]).float()
		next_observations = torch.stack([torch.tensor(entry.next_observation) for entry in log_entries]).float()
		actions = torch.tensor([entry.action for entry in log_entries]).unsqueeze(1)
		rewards = torch.tensor([entry.reward for entry in log_entries])
		return observations, next_observations, actions, rewards

	def trim(self):
		if self.__max_size != None and len(self.__log) > self.__max_size:
			self.__log = self.__log[len(self.__log) - self.__max_size:]

	def get_max_size(self):
		return self.__max_size

	def get_log(self):
		return self.__log

	def record_results(self, results):
		pass

	def __len__(self):
		return len(self.__log)

class PrioritizedLog(SimpleLog):

	def __init__(self, max_size=None, epsilon=0.1, weight_coeff=2):
		super().__init__(max_size)
		self.__epsilon = epsilon
		self.__last_log_idxs = []
		self.__last_losses = []
		self.__starter_loss = 1_000_000
		self.__weight_coeff = weight_coeff

	def add(self, current_observation, next_observation, action, reward):
		super().add(current_observation, next_observation, action, reward)
		self.__last_losses.append(self.__starter_loss)

	def trim(self):
		super().trim()
		if super().get_max_size() != None and len(self.__last_losses) > super().get_max_size():
			self.__last_losses = self.__last_losses[len(self.__last_losses) - super().get_max_size():]
			self.__last_log_idxs = []


	def sample_batch(self, batch_size):
		log = super().get_log()
		self.__last_log_idxs = random.choices(range(0, len(log)), k=batch_size, weights=self.__last_losses)
		log_entries = [log[idx] for idx in self.__last_log_idxs]
		return super().construct_batch_from_sample(log_entries)

	def record_results(self, losses):
		for i, idx in enumerate(self.__last_log_idxs):
			self.__last_losses[idx] = (losses[i].item() + self.__epsilon) * self.__weight_coeff
		self.__last_log_idxs = []
