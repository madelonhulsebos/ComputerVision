import tensorflow as tf
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
import os
import time
import re
from skimage import io, util
from recognition.cnn.model import logits, X, Y
from recognition.cnn.config import NUM_EPOCHS, BATCH_SIZE, DEBUG_STEP_SIZE, NUM_CLASSES, CHARACTER_WIDTH, \
    CHARACTER_HEIGHT, CONVERGENCE_THRESHOLD

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def read_image(queue):
    label = tf.one_hot(indices=queue[1],
                       depth=NUM_CLASSES,
                       on_value=1,
                       off_value=0)
    file_contents = tf.read_file(queue[0], )
    sample = tf.reshape(tf.image.decode_jpeg(file_contents, channels=1), shape=(CHARACTER_HEIGHT * CHARACTER_WIDTH, ))

    return sample, label


def get_paths_and_labels(characters_dir):
    character_files = []
    for f in os.listdir(characters_dir):
        if re.match(r'[^.]+\.jpg$', f):
            path = '%s/%s' % (characters_dir, f)
            try:
                io.imread(path)
                character_files.append(f)
            except OSError:
                pass
    character_paths = ['%s/%s' % (characters_dir, filename) for filename in character_files]
    labels = [int(filename.split('_')[0]) for filename in character_files]

    images = tf.convert_to_tensor(character_paths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)

    return images, labels

# Load tensors for the files and corresponding labels that we will be using for training/validation
train_files, train_labels = get_paths_and_labels('../../datasets/characters/noisy_train')
test_files, test_labels = get_paths_and_labels('../../datasets/characters/noisy_test')

# Makes a queue to read training images.
train_queue = tf.train.slice_input_producer([train_files, train_labels],
                                            shuffle=True)

# Makes a queue to read test images.
test_queue = tf.train.slice_input_producer([test_files, test_labels],
                                           shuffle=True)

train_image, train_label = read_image(train_queue)
test_image, test_label = read_image(test_queue)

train_image_batch, train_label_batch = tf.train.shuffle_batch([train_image, train_label],
                                                              batch_size=BATCH_SIZE,
                                                              min_after_dequeue=10,
                                                              capacity=10000 + 3 * BATCH_SIZE,
                                                              shapes=[(CHARACTER_HEIGHT * CHARACTER_WIDTH, ),
                                                                      (NUM_CLASSES, )])

test_image_batch, test_label_batch = tf.train.shuffle_batch([test_image, test_label],
                                                            batch_size=int(test_files.shape[0]),
                                                            min_after_dequeue=BATCH_SIZE,
                                                            capacity=int(test_files.shape[0]) + 3 * BATCH_SIZE,
                                                            shapes=[(CHARACTER_HEIGHT * CHARACTER_WIDTH, ),
                                                                    (NUM_CLASSES, )])

# Define and optimization
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
loss = tf.reduce_mean(entropy, name='loss')
optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

# Validation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Saver
save_dir = './models/%d' % int(time.time())
os.makedirs(save_dir)
save_path = '%s/model' % save_dir
saver = tf.train.Saver(max_to_keep=5, var_list=tf.global_variables())

print('Starting training! Models will be stored in the file %s every %d iterations' % (save_path, DEBUG_STEP_SIZE))

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    total_loss = 0
    for i in range(0, int(int(train_files.shape[0]) / BATCH_SIZE * NUM_EPOCHS)):
        batch_images, batch_labels = sess.run([train_image_batch, train_label_batch])

        _, loss_value = sess.run([optimizer, loss], feed_dict={X: batch_images, Y: batch_labels})

        total_loss += loss_value

        if i > 0 and i % DEBUG_STEP_SIZE == 0:
            print('Average loss at step %d: %0.2f' % (i, total_loss / DEBUG_STEP_SIZE))

            if total_loss < CONVERGENCE_THRESHOLD:
                print('Convergence reached!')
                break

            total_loss = 0

            saver.save(sess, save_path)

    print('Training finished!')

    test_images, test_labels = sess.run([test_image_batch, test_label_batch])
    print('Test accuracy %g' % accuracy.eval(feed_dict={X: test_images, Y: test_labels}))

    coord.request_stop()
    coord.join(threads)
