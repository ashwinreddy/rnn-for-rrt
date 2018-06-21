from .abstract import ExampleGenerator
from numpy.random import randint
import numpy as np
from constants import quadratic_step_size, Nmax

def slope(p1, p2):
    difference  = p2 - p1
    return difference[1] / difference[0]

class QuadraticInterpolationGenerator(ExampleGenerator):
    def __init__(self, Nmax):
        super().__init__(Nmax)

    def _generateData(self, num_examples):
        examples = []

        for _ in range(num_examples):
            # startAndEnd = randint(-10, 11, size=(1, 4))
            startAndEnd = np.array([[0, 0, 5, 5]])
            # steps = randint(2, self.Nmax)
            steps = 2

            start = startAndEnd[0,:2]
            goal = startAndEnd[0,2:]

            midpoint = (start + goal) / 2
            
            bisectorSlope = -1 / slope(start, goal)

            controlPoint = midpoint + -quadratic_step_size * np.array([1, bisectorSlope])
            
            beginningSlope = slope(start, controlPoint)
            endingSlope = slope(controlPoint, goal)

            beginningPoints = []
            endingPoints = [controlPoint]

            if steps % 2 == 0:
                beginningSteps = steps / 2
                endingSteps = steps / 2
            else:
                beginningSteps = steps // 2
                endingSteps = beginningSteps + 1

            for i in range(steps):
                if i % 2 == 0:
                    beginningPoints.append(start + beginningSlope * ((i+1) / beginningSteps ))
                elif i % 2 == 1:
                    endingPoints.append(controlPoint + endingSlope * (i / endingSteps))
            
            endingPoints.append(goal)

            trajectory = np.vstack((beginningPoints, endingPoints))


            print(len(trajectory))
            
            trajectory = np.diff(trajectory.T).T
            
            
            # trajectory = np.hstack((trajectory, np.zeros((3,1)) ))

            # yield np.tile(np.append(startAndEnd, steps), (self.Nmax, 1)), trajectory