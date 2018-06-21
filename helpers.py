import numpy as np
from constants import dataset_directory, Nmax

numSteps = lambda seq: seq[:,2].tolist().index(1)

seqToPath = lambda seq: np.cumsum(np.delete(seq, 2, 1), axis=0)


def processData(name):
	return np.load(dataset_directory(name, "inputs") + ".npy"), np.load(dataset_directory(name, "outputs") + ".npy")

class Converters:
	@staticmethod
	def pathToVectors(path):
		return np.diff(path)
	
	@staticmethod
	def vectorsToPath(vectors):
		return np.cumsum(vectors, axis=0)