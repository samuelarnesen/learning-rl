class StatsCollector:

	def __init__(self, reduction=True, round=True):
		self.__count = 0
		self.__stats = {}
		self.__reduce = reduction
		self.__round = round

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
		def compute(stat):
			num = self.__stats[stat] / self.__count if self.__reduce else self.__stats[stat]
			if self.__round:
				return round(num, 2)
			return num
		print(delimiter.join([stat + ": "+ str(compute(stat)) for stat in self.__stats]))


