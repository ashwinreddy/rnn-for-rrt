import numpy as np
from pprint import pprint as print
from numpy.random import randint
from constants import Nmax, num_examples, dataset_directory
import matplotlib.pyplot as plt
import random
import math
import copy
import sys
from data_gen import LinearInterpolationGenerator




def main(generatorType, Nmax, num_examples):
	if generatorType == "linear":
		trainingDataGenerator = LinearInterpolationGenerator
	elif generatorType == "rrt":
		trainingDataGenerator = RrtGenerator

	trainingDataGenerator(Nmax).makeDataset(num_examples)


if __name__ == "__main__":
	main(sys.argv[1], Nmax, num_examples)