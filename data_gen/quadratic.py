from .abstract import ExampleGenerator
from numpy.random import randint
import numpy as np

class QuadraticInterpolationGenerator(ExampleGenerator):
    def __init__(self, Nmax):
        super().__init__(Nmax)

    def _generateData(self, num_examples):
        examples = []

        for _ in range(num_examples):
            # startAndEnd = randint(-10, 11, size=(1, 4))
            startAndEnd = np.array([[0, 0, 5, 5]])
            # steps = randint(2, self.Nmax)
            steps = 5
            points = np.tile(np.append(startAndEnd, steps), (self.Nmax, 1))
            
            delta = np.array((startAndEnd[0, 2] - startAndEnd[0, 0], startAndEnd[0, 3] - startAndEnd[0, 1])) / steps

            delta = delta + np.array([0.5, 0.5])
            print(np.cumsum(np.array([delta] * steps), axis=0)) 
            # trajectory = np.cumsum(delta, axis=0)

            # print(trajectory)

            # trajectory = np.append(np.array([delta / steps] * steps), [(0, 0, 1)] * (self.Nmax - steps)).reshape(self.Nmax, 3)

            # trajectory[-1, 2] = 1

            yield points, trajectory