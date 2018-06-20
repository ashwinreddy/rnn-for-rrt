import numpy as np
from pprint import pprint as print
from numpy.random import randint
from constants import Nmax, num_examples
import matplotlib.pyplot as plt
import random
import math
import copy


class RRT():
	def __init__(self, start, goal, obstacleList,
				 randArea, expandDis=1.0, goalSampleRate=5, maxIter=500):
		"""
		Setting Parameter

		start: Start Position [x,y]
		goal: Goal Position [x,y]
		obstacleList: obstacle Positions [[x,y,size],...]
		randArea: Random Samping Area [min,max]

		"""
		self.start = Node(start[0], start[1])
		self.end = Node(goal[0], goal[1])
		self.minrand = randArea[0]
		self.maxrand = randArea[1]
		self.expandDis = expandDis
		self.goalSampleRate = goalSampleRate
		self.maxIter = maxIter
		self.obstacleList = obstacleList

	def Planning(self, animation=True):
		u"""
		Pathplanning

		animation: flag for animation on or ofshow_animation = True
		"""

		self.nodeList = [self.start]
		while True:
			# Random Sampling
			if random.randint(0, 100) > self.goalSampleRate:
				rnd = [random.uniform(self.minrand, self.maxrand), random.uniform(
					self.minrand, self.maxrand)]
			else:
				rnd = [self.end.x, self.end.y]

			# Find nearest node
			nind = self.GetNearestListIndex(self.nodeList, rnd)
			# print(nind)

			# expand tree
			nearestNode = self.nodeList[nind]
			theta = math.atan2(rnd[1] - nearestNode.y, rnd[0] - nearestNode.x)

			newNode = copy.deepcopy(nearestNode)
			newNode.x += self.expandDis * math.cos(theta)
			newNode.y += self.expandDis * math.sin(theta)
			newNode.parent = nind

			if not self.__CollisionCheck(newNode, self.obstacleList):
				continue

			self.nodeList.append(newNode)

			# check goal
			dx = newNode.x - self.end.x
			dy = newNode.y - self.end.y
			d = math.sqrt(dx * dx + dy * dy)
			if d <= self.expandDis:
				break

			if animation:
				self.DrawGraph(rnd)

		path = [[self.end.x, self.end.y]]
		lastIndex = len(self.nodeList) - 1
		while self.nodeList[lastIndex].parent is not None:
			node = self.nodeList[lastIndex]
			path.append([node.x, node.y])
			lastIndex = node.parent

		path.append([self.start.x, self.start.y])

		return path

	def DrawGraph(self, rnd=None):
		u"""
		Draw Graph
		"""
		plt.clf()
		if rnd is not None:
			plt.plot(rnd[0], rnd[1], "^k")
		for node in self.nodeList:
			if node.parent is not None:
				plt.plot([node.x, self.nodeList[node.parent].x], [
					node.y, self.nodeList[node.parent].y], "-g")

		for (ox, oy, size) in self.obstacleList:
			plt.plot(ox, oy, "ok", ms=30 * size)

		plt.plot(self.start.x, self.start.y, "xr")
		plt.plot(self.end.x, self.end.y, "xr")
		plt.axis([-2, 15, -2, 15])
		plt.grid(True)
		plt.pause(0.01)

	def GetNearestListIndex(self, nodeList, rnd):
		dlist = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1])
				 ** 2 for node in nodeList]
		minind = dlist.index(min(dlist))
		return minind

	def __CollisionCheck(self, node, obstacleList):

		for (ox, oy, size) in obstacleList:
			dx = ox - node.x
			dy = oy - node.y
			d = math.sqrt(dx * dx + dy * dy)
			if d <= size:
				return False  # collision

		return True  # safe


class Node():
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.parent = None

def linearGenerateExamples(Nmax, numExamples):
	examples = []

	for _ in range(numExamples):
		startAndEnd = randint(-10, 11, size=(1, 4))
		steps = randint(2, Nmax)
		# steps = Nmax
		points = np.tile(np.append(startAndEnd, steps), (Nmax, 1))

		delta = np.array((startAndEnd[0, 2] - startAndEnd[0, 0], startAndEnd[0, 3] - startAndEnd[0, 1], 0))

		trajectory = np.append(np.array([delta / steps] * steps), [(0, 0, 1)] * (Nmax - steps)).reshape(Nmax, 3)

		trajectory[-1, 2] = 1

		yield points, trajectory

def rrtGenerateExamples(Nmax, num_examples):
	genPoint = lambda: randint(-10, 11, size=(2,))
	obstacleList = [
		
	]

	for i in range(num_examples):

		startAndEnd = randint(-10, 11, size=(1, 4))

		# startAndEnd = np.array([0,0, 5,5])

		rrt = RRT(start=[startAndEnd[0, 0], startAndEnd[0, 1]], goal=[startAndEnd[0, 2], startAndEnd[0, 3]],
				  randArea=[-2, 10], obstacleList=obstacleList)
		path = np.array(rrt.Planning(animation=False))
		delta = np.diff(np.flip(path, 0).T).T
		steps = len(delta)

		points = np.tile(np.append(startAndEnd, steps), (Nmax, 1))
		trajectory = np.c_[delta, np.zeros((steps, 1))]
		zeroed_out = np.tile([0, 0, 1], (Nmax - steps, 1))
		trajectory = np.vstack((trajectory, zeroed_out))

		if i % 500 == 0:
			print(i)

		yield points.tolist(), trajectory.tolist()

def saveExamples(Nmax, numExamples, name, trainingDataGenerator):
	points = []
	trajectories = []
	for pi, ti in trainingDataGenerator(Nmax, numExamples):
		points.append(pi)
		trajectories.append(ti)

	print("Saving {} examples".format(name))
	np.save("./{}-data-inputs".format(name), points)
	np.save("./{}-data-outputs".format(name), trajectories)


def main(generatorType):
	if generatorType == "linear":
		trainingDataGenerator = linearGenerateExamples
	elif generatorType == "rrt":
		trainingDataGenerator = rrtGenerateExamples

	saveExamples(Nmax, num_examples[0], "training", trainingDataGenerator)
	saveExamples(Nmax, num_examples[1], "validation", trainingDataGenerator)
	saveExamples(Nmax, num_examples[2], "testing", trainingDataGenerator)


if __name__ == "__main__":
	# rrtSaveExamples(10_000, "training")
	# rrtSaveExamples(200, "validation")
	# rrtSaveExamples(200, "testing")
	main()
# rrtSaveExamples(2, "training")