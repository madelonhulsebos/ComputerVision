import tensorflow as tf
from skimage import io
from recognition.cnn.model import logits, X, Y
from recognition.cnn.config import NUM_EPOCHS, BATCH_SIZE, DEBUG_STEP_SIZE, NUM_CLASSES, CHARACTER_WIDTH, \
    CHARACTER_HEIGHT, CONVERGENCE_THRESHOLD
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CHARACTERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
              'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def classify_characters(characters):
    predictions = []
    classification = tf.argmax(logits, 1)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        saver.restore(sess, 'recognition/cnn/models/model-1498673617/model')

        for c in characters:
            result = np.reshape(c, (-1, CHARACTER_WIDTH * CHARACTER_HEIGHT))
            prediction = sess.run([classification], feed_dict={X: result, Y: np.zeros((1, NUM_CLASSES))})

            predictions.append(CHARACTERS[int(prediction[0][0])])
    return predictions
