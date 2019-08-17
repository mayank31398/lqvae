import os
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score

from Classifier import Classifier
from LQVAE import LQVAE
from params import (BATCH_SIZE, DATA_PATH, IMAGE_PATH, LATENT_DIM,
                    LEARN_RATE_INIT_CLASSIFIER, LEARN_RATE_INIT_LQVAE,
                    LOG_EVERY, MAX_ITERS_CLASSIFIER, MAX_ITERS_LQVAE,
                    MODEL_PATH, SAVE_EVERY, TEST_EVERY, FGSM_IMAGE_PATH)
from utils import LoadData, SaveImages

warnings.simplefilter("ignore")
tf.logging.set_verbosity(tf.logging.ERROR)


def Evaluate(lqvae, classifier, X, y, session):
    batchNum = 0
    NumBatches = X.shape[0] // BATCH_SIZE

    ground_truths = []
    predictions_raw = []
    predictions_continuous = []
    predictions_discrete = []
    while batchNum < NumBatches:
        x_ = X[batchNum * BATCH_SIZE: (batchNum + 1) * BATCH_SIZE]
        y_ = y[batchNum * BATCH_SIZE: (batchNum + 1) * BATCH_SIZE]

        batchNum += 1
        # Ground truth labels
        y_1 = np.argmax(y_, axis=1)
        ground_truths.append(y_1)

        # Predicted labels on raw images
        pred = classifier.forward(x_, session)
        pred1 = np.argmax(pred, axis=1)

        accuracy_raw = accuracy_score(y_1, pred1)
        predictions_raw.append(pred1)

        # Predicted labels after filtering
        x_1, x_2 = lqvae.forward(x_, session)

        pred = classifier.forward(x_1, session)
        pred1 = np.argmax(pred, axis=1)

        pred = classifier.forward(x_2, session)
        pred2 = np.argmax(pred, axis=1)

        accuracy_continuous = accuracy_score(y_1, pred1)
        accuracy_discrete = accuracy_score(y_1, pred2)

        predictions_continuous.append(pred1)
        predictions_discrete.append(pred2)

        print('step', batchNum)
        print('accuracy raw', accuracy_raw)
        print('accuracy continuous', accuracy_continuous)
        print('accuracy discrete', accuracy_discrete)
        print()

    ground_truths = np.concatenate(ground_truths, axis=0)
    predictions_raw = np.concatenate(predictions_raw, axis=0)
    predictions_continuous = np.concatenate(predictions_continuous, axis=0)
    predictions_discrete = np.concatenate(predictions_discrete, axis=0)

    accuracy_raw = accuracy_score(ground_truths, predictions_raw)
    accuracy_continuous = accuracy_score(ground_truths, predictions_continuous)
    accuracy_discrete = accuracy_score(ground_truths, predictions_discrete)

    print('batch')
    print('accuracy raw', accuracy_raw)
    print('accuracy continuous', accuracy_continuous)
    print('accuracy discrete', accuracy_discrete)


def Bits(lqvae, classifier, X, X_adv, session):
    x1 = tf.placeholder(tf.float32, [None, 28, 28, 1])
    ep1 = tf.random_normal(shape=[BATCH_SIZE, LATENT_DIM])

    z_mean, z_sigm = lqvae.Encoder(x1, reuse=True)
    z_x = z_mean + 0.1 * 0.5 * tf.sqrt(tf.exp(z_sigm)) * ep1
    z_hardsign_x = tf.sign((0.01 - z_x ** 2))

    batchNum = 0
    NumBatches = X.shape[0] // BATCH_SIZE

    l = []
    while batchNum < NumBatches:
        x_ = X[batchNum * BATCH_SIZE: (batchNum + 1) * BATCH_SIZE]
        x_adv = X_adv[batchNum * BATCH_SIZE: (batchNum + 1) * BATCH_SIZE]
        batchNum += 1

        z_2 = session.run(z_hardsign_x, feed_dict={x1: x_})
        z_2_adv = session.run(z_hardsign_x, feed_dict={x1: x_adv})

        bit_flips = (z_2 != z_2_adv).mean()
        l.append(bit_flips)
        print('bit_flips', bit_flips)

    l = np.array(l)
    print("batch bit_flips", l.sum() / l.shape[0])


def FGSM(lqvae, classifier, X, y, session, image_path):
    y1 = tf.placeholder(tf.float32, [None, 10])
    x1 = tf.placeholder(tf.float32, [None, 28, 28, 1])
    ep1 = tf.random_normal(shape=[BATCH_SIZE, LATENT_DIM])

    z_mean, z_sigm = lqvae.Encoder(x1, reuse=True)
    z_x = z_mean + 0.1 * 0.5 * tf.sqrt(tf.exp(z_sigm)) * ep1
    z_softsign_x = tf.nn.softsign((0.01 - z_x ** 2) * 500)
    x3 = lqvae.GeneratorDiscrete(z_softsign_x, reuse=True)

    predictions = classifier.Classify(x3, reuse=True)
    C_loss = tf.losses.sigmoid_cross_entropy(y1, predictions)

    grad = tf.gradients(C_loss, x1)[0]
    grad = tf.sign(grad)

    batchNum = 0
    NumBatches = X.shape[0] // BATCH_SIZE

    FGSM_X = []
    FGSM_y = []
    while batchNum < NumBatches:
        x_ = X[batchNum * BATCH_SIZE: (batchNum + 1) * BATCH_SIZE]
        y_ = y[batchNum * BATCH_SIZE: (batchNum + 1) * BATCH_SIZE]

        # for _ in range(10):
        x_ = x_ + 0.3 * session.run(grad, feed_dict={x1: x_, y1: y_})

        FGSM_X.append(x_)
        FGSM_y.append(y_)

        # SaveImages(x_, image_path, name=str(batchNum))

        batchNum += 1

    FGSM_X = np.concatenate(FGSM_X, axis=0)
    FGSM_y = np.concatenate(FGSM_y, axis=0)

    np.save("Data/X_test_FGSM.npy", FGSM_X)
    np.save("Data/y_test_FGSM.npy", FGSM_y)


def CalculateBitFlips():
    X_adv, _ = LoadData(DATA_PATH, file="npy")
    _, (X_test, _) = LoadData(DATA_PATH)

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

    with tf.Session() as session:
        saver.restore(session, MODEL_PATH)

        Bits(lqvae, classifier, X_test, X_adv, session)


def EvaluatePerformance():
    # _, (X, y) = LoadData(DATA_PATH)
    X, y = LoadData(DATA_PATH, file="npy")

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

    with tf.Session() as session:
        saver.restore(session, MODEL_PATH)

        Evaluate(lqvae, classifier, X, y, session)


def DriveFGSM():
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

    with tf.Session() as session:
        saver.restore(session, MODEL_PATH)

        FGSM(lqvae, classifier, X_test, y_test, session, FGSM_IMAGE_PATH)


if __name__ == "__main__":
    # DriveFGSM()
    EvaluatePerformance()
