import tensorflow as tf
from rnn import SketchRNN
from constants import Nmax, num_coordinates
import matplotlib.pyplot as plt
from helpers import seqToPath
import numpy as np

def test():
	with tf.Session() as sess:
		network = SketchRNN(seq_size=Nmax, num_coordinates=num_coordinates)
		saver = tf.train.import_meta_graph("./saves/model.ckpt.meta")
		saver.restore(sess, tf.train.latest_checkpoint("./saves/"))

		steps = Nmax

		results = network(
			[[[0, 0, 5, 5, steps]] * Nmax]
			# testingInput
		)

		path = np.vstack(([0,0], seqToPath(results[0][:steps])))
		# print(np.linalg.norm(results.sum(1)[0][:-1] - np.array([1, 1])))
		
		plt.plot(path[:, 0], path[:, 1])
		plt.show()
	# # network.close()