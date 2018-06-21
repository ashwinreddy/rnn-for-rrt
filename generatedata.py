import numpy as np
from pprint import pprint as print
from numpy.random import randint
from constants import Nmax, num_examples, dataset_directory
import matplotlib.pyplot as plt
import random
import math
import copy
import sys
from data_gen import LinearInterpolationGenerator, RrtGenerator, QuadraticInterpolationGenerator




def main(generatorType, Nmax, num_examples):
	generatorTypes = {
		"linear": LinearInterpolationGenerator,
		"rrt": RrtGenerator,
		"quadratic": QuadraticInterpolationGenerator
	}
	
	generatorTypes[generatorType](Nmax).makeDataset(num_examples)


if __name__ == "__main__":
	main(sys.argv[1], Nmax, num_examples)