import numpy as np
import tensorflow as tf


class SketchRNN(object):
	def __init__(self, seq_size, num_coordinates=2):
		print("Instantiating network")
		self._setup_network(seq_size, num_coordinates)
		self._setup_cost()
		self._setup_saver()

	def _setup_network(self, seq_size, num_coordinates):
		print("Setting up network")

		self.network_dimensions = (1 + num_coordinates * 2, 1 + num_coordinates)

		self._setup_placeholders(seq_size, num_coordinates)
		self.lstm_layers = [tf.contrib.rnn.BasicLSTMCell(100), tf.contrib.rnn.BasicLSTMCell(100)]
		self.lstm = tf.contrib.rnn.MultiRNNCell(self.lstm_layers)
		self.lstm_bw = tf.contrib.rnn.MultiRNNCell(self.lstm_layers)
		self.rnn_output, _ = tf.nn.dynamic_rnn(self.lstm, self.inputs, dtype="float")
		# print(tf.shape(self.rnn_output))
		self.net_outputs = tf.contrib.layers.fully_connected(self.rnn_output, 100, activation_fn=tf.identity)
		self.net_outputs = tf.contrib.layers.fully_connected(self.net_outputs, self.network_dimensions[1],
															 activation_fn=tf.identity)

	# self.factors = tf.Variable(tf.random_normal([1,2]), name="factors")
	# self.net_outputs = tf.multiply(self.net_outputs,  self.factors)

	def _setup_placeholders(self, seq_size, num_coordinates):
		print("Creating placeholders")
		self.inputs = tf.placeholder("float", [None, seq_size, self.network_dimensions[0]], name="inputs_placeholder")
		self.outputs = tf.placeholder("float", [None, seq_size, self.network_dimensions[1]], name="outputs_placeholder")

	def _setup_cost(self):
		print("Creating cost function")
		# self.cost = tf.reduce_mean(tf.square( self.net_outputs - self.outputs ))
		self._cost = tf.nn.l2_loss(self.net_outputs - self.outputs)
		tf.summary.scalar('cost', self._cost)
		self.train_op = tf.train.AdamOptimizer().minimize(self._cost)

	def _setup_saver(self):
		print("Creating saver and summary operation")
		self.saver = tf.train.Saver()
		self.merged = tf.summary.merge_all()
		self.session = tf.Session()

		tf.get_variable_scope().reuse_variables()
		self.session.run(tf.global_variables_initializer())

	def train(self, trainingData, validationData, iterations):
		print("Training network on examples")
		validation_cost = self.cost(validationData[0], validationData[1])
		# with tf.Session() as sess:
		fileWriter = tf.summary.FileWriter("./saves/model.ckpt", self.session.graph)

		#     tf.get_variable_scope().reuse_variables()
		#     sess.run(tf.global_variables_initializer())

		for i in range(iterations):
			summary, _, error = self.session.run([self.merged, self.train_op, self._cost], feed_dict={
				self.inputs: trainingData[0],
				self.outputs: trainingData[1]
			})

			fileWriter.add_summary(summary, i)

			if i % 350 == 0:
				print(i, error)

				new_validation_cost = self.cost(validationData[0], validationData[1])

				if new_validation_cost >= validation_cost:
					break
				else:
					self.saver.save(self.session, "./saves/model.ckpt")
					validation_cost = new_validation_cost
		self.saver.save(self.session, "./saves/model.ckpt")

	def restore(self):
		self.saver.restore(self.session, './saves/model.ckpt')

	def __call__(self, inputSequence):
		print("Calling network on input")
		self.restore()
		return self.session.run(self.net_outputs, feed_dict={self.inputs: inputSequence})

	def cost(self, inputData, outputAnswers):
		return self.session.run(self._cost, feed_dict={
			self.inputs: inputData,
			self.outputs: outputAnswers
		})

	def close(self):
		self.session.close()