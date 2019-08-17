import os
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

from Classifier import Classifier
from LQVAE import LQVAE
from params import (BATCH_SIZE, DATA_PATH, IMAGE_PATH, LATENT_DIM,
                    LEARN_RATE_INIT_CLASSIFIER, LEARN_RATE_INIT_LQVAE,
                    LOG_EVERY, MAX_ITERS_CLASSIFIER, MAX_ITERS_LQVAE,
                    MODEL_PATH, SAVE_EVERY, TEST_EVERY)
from utils import LoadData

warnings.simplefilter("ignore")
tf.logging.set_verbosity(tf.logging.ERROR)


def main():
    (X_train, y_train), (X_test, y_test) = LoadData(DATA_PATH)

    lqvae = LQVAE(batch_size=BATCH_SIZE,
                  max_iters=MAX_ITERS_LQVAE,
                  latent_dim=LATENT_DIM,
                  learnrate_init=LEARN_RATE_INIT_LQVAE,
                  log_every=LOG_EVERY,
                  save_every=SAVE_EVERY,
                  image_path=IMAGE_PATH)
    classifier = Classifier(batch_size=BATCH_SIZE,
                            max_iters=MAX_ITERS_CLASSIFIER,
                            learnrate_init=LEARN_RATE_INIT_CLASSIFIER,
                            log_every=LOG_EVERY,
                            test_every=TEST_EVERY)

    lqvae.build_model_lqvae()
    classifier.build_model_classifier()

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)

        lqvae.train(X_train, session)
        classifier.train(X_train, y_train, X_test, y_test, session)

        saver.save(session, MODEL_PATH)


if __name__ == "__main__":
    main()
