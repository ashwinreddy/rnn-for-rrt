import numpy as np
import tensorflow as tf
from constants import Nmax, training_iterations, num_coordinates, dataset_directory
from helpers import processData
from rnn import SketchRNN



def trainNetwork():
	network = SketchRNN(seq_size=Nmax, num_coordinates=num_coordinates)

	trainingInput, trainingAnswers = processData("training")
	validationInput, validationAnswers = processData("validation")

	# print(trainingInput)
	# print("*******************")
	# print(trainingAnswers)

	network.train([trainingInput, trainingAnswers], [validationInput, validationAnswers], training_iterations)

	testingInput, testingAnswers = processData("testing")
	print(network.cost(testingInput, testingAnswers))

	network.close()