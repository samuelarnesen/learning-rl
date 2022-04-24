class StatsCollector:

	def __init__(self):
		self.__count = 0
		self.__stats = {}

	def reset(self):
		self.__count = 0
		self.__stats = {}

	def add(self, stat_collection):
		self.__count += 1
		for stat in stat_collection:
			if stat not in self.__stats:
				self.__stats[stat] = 0
			self.__stats[stat] += stat_collection[stat]

	def show(self, delimiter=", "):
		print(delimiter.join([stat + ": "+ str(self.__stats[stat] / self.__count) for stat in self.__stats]))


