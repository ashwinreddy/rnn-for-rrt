from .abstract import ExampleGenerator
import numpy as np
from numpy.random import randint
from constants import quadratic_step_size, Nmax
from .helpers import padSequence
from scipy import interpolate
import matplotlib.pyplot as plt

def slope(p1, p2):
    difference  = p2 - p1
    return difference[1] / difference[0]

class QuadraticInterpolationGenerator(ExampleGenerator):
    def __init__(self, Nmax):
        print("Creating Quadratic Generator")
        super().__init__(Nmax)

    def _generateData(self, num_examples):
        examples = []

        for _ in range(num_examples):
            # endpoints = np.random.rand(1, 4) * 6
            endpoints = np.array([[0,0, np.random.rand() * 6, np.random.rand() * 6]])
            steps = randint(2, self.Nmax)

            start = endpoints[0,:2]
            goal = endpoints[0,2:]

            midpoint = (start + goal) / 2
            
            bisectorSlope = -1 / slope(start, goal)

            controlPoint = midpoint + -quadratic_step_size * np.array([1, bisectorSlope])
            
            # beginningSlope = slope(start, controlPoint)
            # endingSlope = slope(controlPoint, goal)

            trajectory = np.array([ start, controlPoint, goal])


            x = trajectory[:,0]
            y = trajectory[:,1]

            # print(x)

            f = interpolate.interp1d(x, y, kind='quadratic')
            xnew = np.linspace(start[0], goal[0], steps)
            plt.plot(xnew, f(xnew))
            # plt.show()

            trajectory = np.hstack((xnew.reshape((steps, 1)), f(xnew).reshape((steps, 1))))

            # print(np.diff(trajectory, axis=0))

            yield np.tile(np.append(endpoints, steps), (self.Nmax, 1)), padSequence(np.diff(trajectory, axis=0), self.Nmax)