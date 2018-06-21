from .abstract import ExampleGenerator
import numpy as np
from numpy.random import randint
from constants import quadratic_step_size, Nmax
from .helpers import padSequence
from scipy import interpolate
import matplotlib.pyplot as plt

def slope(p1, p2):
    # print(p1, p2)
    difference  = p2 - p1
    return difference[1] / difference[0]

class QuadraticInterpolationGenerator(ExampleGenerator):
    def __init__(self, Nmax):
        print("Creating Quadratic Generator")
        super().__init__(Nmax)

    def _generateData(self, num_examples):
        examples = []

        for _ in range(num_examples):
            # endpoints = randint(0, 5, size=(1,4))
            endpoints = np.array([[2, 1, 5, 5]])
            steps = randint(2, self.Nmax)

            start = endpoints[0,:2]
            goal = endpoints[0,2:]

            midpoint = (start + goal) / 2
            
            bisectorSlope = -1 / slope(start, goal)

            controlPoint = midpoint + -quadratic_step_size * np.array([1, bisectorSlope])
            
            # beginningSlope = slope(start, controlPoint)
            # endingSlope = slope(controlPoint, goal)

            trajectory = np.array([ start, controlPoint - start, goal - controlPoint])

            # trajectory = padSequence(trajectory, self.Nmax)

            trajectory =   np.cumsum(trajectory, axis=0) 

            x = trajectory[:,0]
            y = trajectory[:,1]

            f = interpolate.interp1d(x, y, kind='quadratic')
            xnew = np.linspace(start[0], goal[0], steps)
            plt.plot(xnew, f(xnew))
            plt.show()

            break

            # yield np.tile(np.append(endpoints, steps), (self.Nmax, 1)), trajectory