from .abstract import ExampleGenerator

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


class RrtGenerator(ExampleGenerator):
	def __init__(self, Nmax):
		super().__init__(Nmax)
	
	def dataGenerator(self, num_examples):
		genPoint = lambda: randint(-10, 11, size=(2,))
		obstacleList = [
			
		]

		for i in range(num_examples):

			endpoints = randint(-10, 11, size=(1, 4))

			# endpoints = np.array([0,0, 5,5])

			rrt = RRT(start=[endpoints[0, 0], endpoints[0, 1]], goal=[endpoints[0, 2], endpoints[0, 3]],
					randArea=[-2, 10], obstacleList=obstacleList)
			path = np.array(rrt.Planning(animation=False))
			delta = np.diff(np.flip(path, 0).T).T
			steps = len(delta)

			points = np.tile(np.append(endpoints, steps), (Nmax, 1))
			trajectory = np.c_[delta, np.zeros((steps, 1))]
			zeroed_out = np.tile([0, 0, 1], (Nmax - steps, 1))
			trajectory = np.vstack((trajectory, zeroed_out))

			if i % 500 == 0:
				print(i)

			yield points.tolist(), trajectory.tolist()