import numpy as np
import tensorflow as tf
from constants import Nmax, training_iterations, num_coordinates

from rnn import SketchRNN


def processData(name):
	outputLabels = np.load("./{}-data-outputs.npy".format(name))
	return np.load("./{}-data-inputs.npy".format(name)), outputLabels


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