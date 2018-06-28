import tensorflow as tf

class PathParamToLatentVecModel(object):
    def __init__(self):
        self.input = tf.placeholder("float", shape=(None, 5))
    
    def __call__(self, input):
        self.model = tf.contrib.layers.fully_connected(self.input, num_units=100, activation=tf.identity)
        self.model = tf.contrib.layers.fully_connected(self.model, num_units=128, activation=tf.identity)
        return self.model

    def train(self, input, true_latent_vectors):
        self.answers = tf.placeholder("float", shape=(None, 128))
        loss = tf.nn.l2_loss(self.call(input) - self.answers)
        train_step = tf.train.AdamOptimizer().minimize(loss)

        with tf.Session() as sess:
            for i in range(1000):
                sess.run(train_step, feed_dict={
                    self.input: input,
                    self.answers: true_latent_vectors
                })
    
model = PathParamToLatentVecModel()

model.train(path_params, latent_vectors)