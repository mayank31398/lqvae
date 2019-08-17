import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score


class Classifier:
    def __init__(self,
                 batch_size,
                 max_iters,
                 learnrate_init,
                 log_every,
                 test_every):
        # Model parameters
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.learn_rate_init = learnrate_init
        self.log_every = log_every
        self.test_every = test_every
        self.channel = 1
        self.input_size = 28
        self.num_classes = 10

        # Create placeholders
        self.x = tf.placeholder(
            tf.float32, [None, self.input_size, self.input_size, self.channel])
        self.y = tf.placeholder(
            tf.float32, [None, self.num_classes])

    def build_model_classifier(self):
        # Set pipeline
        self.predictions = self.Classify(self.x)
        self.C_loss = tf.losses.sigmoid_cross_entropy(self.y, self.predictions)
        self.cl_vars = tf.trainable_variables(scope='classifier')

        # Learning rate decay
        global_step = tf.Variable(0, trainable=False)
        self.add_global = global_step.assign_add(1)
        self.new_learning_rate = tf.train.exponential_decay(
            self.learn_rate_init, global_step=global_step, decay_steps=10000, decay_rate=0.98)

        # Optimizer
        self.opti_c = tf.train.RMSPropOptimizer(
            learning_rate=self.new_learning_rate).minimize(self.C_loss, var_list=self.cl_vars)

    def Classify(self, x, reuse=False):
        with tf.variable_scope('classifier', reuse=reuse) as scope:
            x = tf.layers.conv2d(x, kernel_size=3, filters=64,
                                 strides=1, padding="same", activation=tf.nn.leaky_relu)
            x = tf.layers.conv2d(x, kernel_size=3, filters=64,
                                 strides=2, padding="same", activation=tf.nn.leaky_relu)
            x = tf.layers.conv2d(x, kernel_size=3, filters=128,
                                 strides=1, padding="same", activation=tf.nn.leaky_relu)
            x = tf.layers.conv2d(x, kernel_size=3, filters=64,
                                 strides=2, padding="same", activation=tf.nn.leaky_relu)
            x = tf.layers.flatten(x)
            x = tf.layers.dense(x, 128 * 7 * 7, activation=tf.nn.leaky_relu)
            x = tf.layers.dense(x, 1024, activation=tf.nn.leaky_relu)
            x = tf.layers.dense(x, self.num_classes,
                                activation=tf.nn.leaky_relu)

        return x

    def train(self, X_train, y_train, X_test, y_test, session):
        step = 0
        batchNum = 0
        NumBatches = X_train.shape[0] // self.batch_size

        print('======================================================================')
        print('Classifier Training Begins')

        while step <= self.max_iters:
            next_x_images = X_train[batchNum *
                                    self.batch_size: (batchNum + 1) * self.batch_size]
            next_y_labels = y_train[batchNum *
                                    self.batch_size: (batchNum + 1) * self.batch_size]

            batchNum = (batchNum + 1) % (NumBatches)
            step += 1
            new_learn_rate = session.run(self.new_learning_rate)

            if new_learn_rate > 0.00005:
                session.run(self.add_global)

            fd = {self.x: next_x_images, self.y: next_y_labels}
            session.run(self.opti_c, feed_dict=fd)

            if step % self.log_every == 0:
                loss, prediction = session.run(
                    [self.C_loss, self.predictions], feed_dict=fd)

                prediction = np.argmax(prediction, axis=1)
                y = np.argmax(next_y_labels, axis=1)
                accuracy = accuracy_score(y, prediction)

                print('step', step)
                print('loss: ', loss)
                print('accuracy', accuracy)

            if step % self.test_every == 0:
                self.test(X_test, y_test, session)
                print('Classifier Training Begins')

    def test(self, X_test, y_test, session):
        print('======================================================================')
        print('Classifier Testing Begins')

        batchNum = 0
        NumBatches = X_test.shape[0] // self.batch_size

        ground_truths = []
        predictions = []
        while batchNum <= NumBatches:
            next_x_images = X_test[batchNum *
                                   self.batch_size: (batchNum + 1) * self.batch_size]
            next_y_labels = y_test[batchNum *
                                   self.batch_size: (batchNum + 1) * self.batch_size]
            batchNum += 1

            fd = {self.x: next_x_images, self.y: next_y_labels}
            loss, prediction = session.run(
                [self.C_loss, self.predictions], feed_dict=fd)

            prediction = np.argmax(prediction, axis=1)
            y = np.argmax(next_y_labels, axis=1)
            accuracy = accuracy_score(y, prediction)

            predictions.append(prediction)
            ground_truths.append(y)

            print('step', batchNum)
            print('loss', loss)
            print('accuracy', accuracy)
            print()

        ground_truths = np.concatenate(ground_truths, axis=0)
        predictions = np.concatenate(predictions, axis=0)
        accuracy = accuracy_score(ground_truths, predictions)
        print('batch_accuracy', accuracy)
        print('======================================================================\n')

    def forward(self, x, session):
        y = session.run(self.predictions, feed_dict={self.x: x})
        return y
