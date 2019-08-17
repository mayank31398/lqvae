import tensorflow as tf
from utils import SaveImages


class LQVAE:
    def __init__(self,
                 batch_size,
                 max_iters,
                 latent_dim,
                 learnrate_init,
                 log_every,
                 save_every,
                 image_path):
        # Model parameters
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.latent_dim = latent_dim
        self.learn_rate_init = learnrate_init
        self.log_every = log_every
        self.save_every = save_every
        self.image_path = image_path
        self.channel = 1
        self.output_size = 28
        self.quant_thresh = 0.1

        # Create placeholders
        self.x_input = tf.placeholder(
            tf.float32, [None, self.output_size, self.output_size, self.channel])
        self.x_true = tf.placeholder(
            tf.float32, [None, self.output_size, self.output_size, self.channel])

        self.ep1 = tf.random_normal(shape=[self.batch_size, self.latent_dim])

    def build_model_lqvae(self):
        # Set the pipeline
        self.z_mean, self.z_sigm = self.Encoder(self.x_input)
        self.z_x = self.z_mean + self.quant_thresh * \
            0.5 * tf.sqrt(tf.exp(self.z_sigm)) * self.ep1

        self.z_softsign_x = tf.nn.softsign(
            (self.quant_thresh ** 2 - self.z_x ** 2) * 500)
        self.z_hardsign_x = tf.sign(self.quant_thresh ** 2 - self.z_x ** 2)

        self.x1 = self.GeneratorContinuous(self.z_softsign_x, reuse=False)
        self.x2 = self.GeneratorDiscrete(self.z_hardsign_x, reuse=False)

        self.z_filt_x = tf.sign(self.quant_thresh ** 2 - self.z_mean ** 2)
        self.x_filt = self.GeneratorDiscrete(self.z_filt_x, reuse=True)

        # Calculate losses
        self.kl_loss = self.KL_loss(
            self.z_mean, self.z_sigm) / (self.latent_dim * self.batch_size)

        self.x1_mse = tf.reduce_mean((self.x1 - self.x_true) ** 2)
        self.x2_mse = tf.reduce_mean((self.x2 - self.x_true) ** 2)
        self.grad = tf.gradients(self.z_mean, self.x_input)[0]
        self.EncoderGradPenality = tf.reduce_mean(
            (0.1 - tf.sqrt(tf.reduce_sum(self.grad ** 2, [1, 2, 3]))) ** 2)

        self.en_loss = self.kl_loss + 10 * self.x1_mse + 10 * self.EncoderGradPenality
        self.gc_loss = 10 * self.x1_mse
        self.gd_loss = 10 * self.x2_mse

        # Train variables
        self.en_vars = tf.trainable_variables(scope="encoder")
        self.gc_vars = tf.trainable_variables(scope="generator_continuous")
        self.gd_vars = tf.trainable_variables(scope="generator_discrete")

        # Learning rate decay
        global_step = tf.Variable(0, trainable=False)
        self.add_global = global_step.assign_add(1)
        self.new_learning_rate = tf.train.exponential_decay(
            self.learn_rate_init, global_step=global_step, decay_steps=10000, decay_rate=0.98)

        # Optimizers
        trainer_gc = tf.train.RMSPropOptimizer(
            learning_rate=self.new_learning_rate)
        gradients_gc = trainer_gc.compute_gradients(
            self.gc_loss, var_list=self.gc_vars)
        self.opti_gc = trainer_gc.apply_gradients(gradients_gc)

        trainer_gd = tf.train.RMSPropOptimizer(
            learning_rate=self.new_learning_rate)
        gradients_gd = trainer_gd.compute_gradients(
            self.gd_loss, var_list=self.gd_vars)
        self.opti_gd = trainer_gd.apply_gradients(gradients_gd)

        trainer_en = tf.train.RMSPropOptimizer(
            learning_rate=self.new_learning_rate)
        gradients_en = trainer_en.compute_gradients(
            self.en_loss, var_list=self.en_vars)
        self.opti_en = trainer_en.apply_gradients(gradients_en)

    def Encoder(self, x, reuse=False):
        with tf.variable_scope('encoder', reuse=reuse) as scope:
            x = tf.layers.conv2d(x, kernel_size=3, filters=64,
                                 strides=1, padding="same", activation=tf.nn.leaky_relu)
            x = tf.layers.conv2d(x, kernel_size=3, filters=64,
                                 strides=2, padding="same", activation=tf.nn.leaky_relu)
            x = tf.layers.conv2d(x, kernel_size=3, filters=128,
                                 strides=2, padding="same", activation=tf.nn.leaky_relu)
            x = tf.layers.conv2d(x, kernel_size=3, filters=128,
                                 strides=1, padding="same", activation=tf.nn.leaky_relu)
            x = tf.layers.flatten(x)
            x = tf.layers.dense(x, 1024, activation=tf.nn.leaky_relu)

            mu = tf.layers.dense(x, self.latent_dim)
            log_sigma = tf.layers.dense(x, self.latent_dim)

        return mu, log_sigma

    def GeneratorContinuous(self, x, reuse=False):
        with tf.variable_scope('generator_continuous', reuse=reuse) as scope:
            x = tf.layers.dense(x, 1024, activation=tf.nn.leaky_relu)
            x = tf.layers.dense(x, 128 * 7 * 7, activation=tf.nn.leaky_relu)
            x = tf.reshape(x, [-1, 7, 7, 128])
            x = tf.layers.conv2d_transpose(
                x, kernel_size=3, filters=128, strides=1, padding="same", activation=tf.nn.leaky_relu)
            x = tf.layers.conv2d_transpose(
                x, kernel_size=3, filters=128, strides=2, padding="same", activation=tf.nn.leaky_relu)
            x = tf.layers.conv2d_transpose(
                x, kernel_size=3, filters=64, strides=2, padding="same", activation=tf.nn.leaky_relu)
            x = tf.layers.conv2d_transpose(
                x, kernel_size=3, filters=1, strides=1, padding="same", activation=tf.nn.tanh)

        return x

    def GeneratorDiscrete(self, x, reuse=False):
        with tf.variable_scope('generator_discrete', reuse=reuse) as scope:
            x = tf.layers.dense(x, 1024, activation=tf.nn.leaky_relu)
            x = tf.layers.dense(x, 128 * 7 * 7, activation=tf.nn.leaky_relu)
            x = tf.reshape(x, [-1, 7, 7, 128])
            x = tf.layers.conv2d_transpose(
                x, kernel_size=3, filters=128, strides=1, padding="same", activation=tf.nn.leaky_relu)
            x = tf.layers.conv2d_transpose(
                x, kernel_size=3, filters=128, strides=2, padding="same", activation=tf.nn.leaky_relu)
            x = tf.layers.conv2d_transpose(
                x, kernel_size=3, filters=64, strides=2, padding="same", activation=tf.nn.leaky_relu)
            x = tf.layers.conv2d_transpose(
                x, kernel_size=3, filters=1, strides=1, padding="same", activation=tf.nn.tanh)

        return x

    def KL_loss(self, mu, log_var):
        return -0.5 * tf.reduce_sum(1 + log_var - tf.pow(mu, 2) - tf.exp(log_var))

    def train(self, X_train, session):
        print('******************')
        print('LQ-VAE Training Begins')
        print('******************')

        step = 0
        batchNum = 0
        NumBatches = X_train.shape[0] // self.batch_size

        while step <= self.max_iters:
            next_x_images = X_train[batchNum *
                                    self.batch_size: (batchNum + 1) * self.batch_size]
            batchNum = (batchNum + 1) % (NumBatches)
            new_learn_rate = session.run(self.new_learning_rate)

            if new_learn_rate > 0.00005:
                session.run(self.add_global)

            fd = {self.x_input: next_x_images, self.x_true: next_x_images}
            session.run(self.opti_en, feed_dict=fd)
            session.run(self.opti_gc, feed_dict=fd)
            session.run(self.opti_gd, feed_dict=fd)

            if step % self.log_every == 0:
                k1_loss, enc_loss, x1_mse, x2_mse = session.run(
                    [self.kl_loss, self.en_loss, self.x1_mse, self.x2_mse], feed_dict=fd)

                print('step', step)
                print('lr:', new_learn_rate)
                print('KL_Loss: ', k1_loss)
                print('Encoder Loss: ', enc_loss)
                print('x1_mse: ', x1_mse)
                print('x2_mse: ', x2_mse)
                print()

            if step % self.save_every == 0:
                x1, x2 = session.run([self.x1, self.x2], feed_dict=fd)

                SaveImages(next_x_images, self.image_path,
                           name=str(step) + "_true")
                SaveImages(x1, self.image_path, name=str(step) + "_continuous")
                SaveImages(x2, self.image_path, name=str(step) + "_discrete")

            step += 1

    def forward(self, x, session):
        fd = {self.x_input: x, self.x_true: x}
        x1, x2 = session.run([self.x1, self.x2], feed_dict=fd)

        return x1, x2
