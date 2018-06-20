from .abstract import ExampleGenerator
from numpy.random import randint
import numpy as np

class LinearInterpolationGenerator(ExampleGenerator):
	def __init__(self, Nmax):
		super().__init__(Nmax)

	def _generateData(self, num_examples):
		examples = []

		for _ in range(num_examples):
			startAndEnd = randint(-10, 11, size=(1, 4))
			steps = randint(2, self.Nmax)
			# steps = Nmax
			points = np.tile(np.append(startAndEnd, steps), (self.Nmax, 1))

			delta = np.array((startAndEnd[0, 2] - startAndEnd[0, 0], startAndEnd[0, 3] - startAndEnd[0, 1], 0))

			trajectory = np.append(np.array([delta / steps] * steps), [(0, 0, 1)] * (self.Nmax - steps)).reshape(self.Nmax, 3)

			trajectory[-1, 2] = 1

			yield points, trajectory