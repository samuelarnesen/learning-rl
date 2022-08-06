import random, math, sys, heapq, functools, pickle, os
import torch

@functools.total_ordering
class LogEntry:

	def __init__(self, current_observation, next_observation, action, reward, done):
		self.current_observation = current_observation
		self.next_observation = next_observation
		self.action = action
		self.reward = reward
		self.done = done

	def __str__(self):
		return " ".join(["Current: ", str(self.current_observation), "\nNext:", str(self.next_observation), "\nReward:", str(self.reward), "\nDone:", str(self.done), "\n"])

	def __repr__(self):
		return " ".join(["Current: ", str(self.current_observation), "\nNext:", str(self.next_observation), "\nReward:", str(self.reward), "\nDone:", str(self.done), "\n"])

	# we just need this for the priority queue and we can break ties arbitrarily
	def __eq__(self, other):
		return random.random() < 0.5

	def __lt__(self, other):
		return random.random() < 0.5

class Log:

	def __init__(self):
		pass

	def add(self, current_observation, next_observation, action, reward, done):
		pass

	def sample(self):
		pass

	def sample_batch(self, batch_size):
		pass

	def record_results(self, losses):
		pass


class SimpleLog(Log):

	def __init__(self, max_size=None, auto_trim=True):
		super().__init__()
		self.__log = []
		self.__max_size = max_size
		self.__auto_trim = auto_trim and max_size != None

	def add(self, current_observation, next_observation, action, reward, done):
		self.__log.append(LogEntry(current_observation, next_observation, action, reward, done))
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

		incompletions = torch.tensor([(0 if entry.done else 1) for entry in log_entries]).unsqueeze(1).float()

		return observations, next_observations, actions, rewards, incompletions

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

class PriorityLog(Log):

	def __init__(self, max_size=None, auto_trim=True, prob=0.5):
		super().__init__()
		self.__log = []
		self.__max_size = max_size
		self.__auto_trim = auto_trim and max_size != None
		self.__prob = prob
		self.__last_sampled = []

	def add(self, current_observation, next_observation, action, reward, done, priority=None):
		heapq.heappush(self.__log, (self.get_priority(priority), LogEntry(current_observation, next_observation, action, reward, done)))
		if self.__max_size != None and len(self.__log) > self.__max_size:
			self.__log = self.__log[int((len(self.__log) / 2)):]

	def sample_batch(self, batch_size):
		log_entries = []
		discarded_entries = []
		while len(log_entries) < batch_size and len(self.__log) > 0:
			entry = heapq.heappop(self.__log)
			if random.random() < self.__prob:
				log_entries.append(entry[-1])
				self.__last_sampled.append(entry[-1])
			else:
				discarded_entries.append(entry)

		for entry in discarded_entries:
			heapq.heappush(self.__log, entry)

		return self.construct_batch_from_sample(log_entries)

	def record_results(self, losses):
		for loss, entry in zip(losses, self.__last_sampled):
			heapq.heappush(self.__log, (-loss.item(), entry))
		self.__last_sampled = []

	def construct_batch_from_sample(self, log_entries):
		observations = torch.stack([torch.tensor(entry.current_observation).squeeze() for entry in log_entries]).float()
		next_observations = torch.stack([torch.tensor(entry.next_observation).squeeze() for entry in log_entries]).float()

		actions = torch.tensor([entry.action for entry in log_entries]).unsqueeze(1)
		rewards = torch.tensor([entry.reward for entry in log_entries])

		incompletions = torch.tensor([(0 if entry.done else 1) for entry in log_entries]).unsqueeze(1).float()

		return observations, next_observations, actions, rewards, incompletions

	def get_max_size(self):
		return self.__max_size

	def get_log(self):
		return self.__log

	def reset(self):
		self.__log = []

	def get_priority(self, priority):
		return random.random() - 1_000_000 if priority == None else priority

	def __len__(self):
		return len(self.__log)

def save_dataset(filepath, dataset):
	with open(filepath, "wb") as f:
		pickle.dump(dataset, f)

def load_dataset(filepath, default_max_size=1_000, default_use_priority=True):
	if os.path.isfile(filepath):
		with open(filepath, "rb") as f:
			return pickle.load(f)
	elif default_use_priority:
		return PriorityLog(max_size=MAX_LOG_SIZE)
	else:
		return SimpleLog(max_size=MAX_LOG_SIZE)

