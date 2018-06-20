import numpy as np
from helpers import dataset_directory

class ExampleGenerator(object):
	def __init__(self, Nmax):
		self.Nmax = Nmax
	
	def _generateData(self, num_examples):
		raise NotImplementedError

	def makeDataset(self, datasetSize):
		for setname, num_examples in [("training", datasetSize[0]), ("validation", datasetSize[1]), ("testing", datasetSize[2])]:
			points = []
			trajectories = []
			for pi, ti in self._generateData(num_examples):
				points.append(pi)
				trajectories.append(ti)

			print("Saving {} examples".format(setname))
			np.save(dataset_directory(setname, "inputs"), points)
			np.save(dataset_directory(setname, "outputs"), trajectories)