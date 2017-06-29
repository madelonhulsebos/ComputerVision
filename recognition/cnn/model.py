import tensorflow as tf
from recognition.cnn.config import NUM_EPOCHS, BATCH_SIZE, DEBUG_STEP_SIZE, NUM_CLASSES, CHARACTER_WIDTH, \
    CHARACTER_HEIGHT, CONVERGENCE_THRESHOLD


# dropout = tf.placeholder(dtype=tf.float32, name='dropout')
# global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

with tf.name_scope('data'):
    X = tf.placeholder(dtype=tf.float32, shape=[None, CHARACTER_HEIGHT * CHARACTER_WIDTH], name='X_placeholder')
    Y = tf.placeholder(dtype=tf.float32, shape=[None, NUM_CLASSES], name='Y_placeholder')

# Input 24 x 44 x 1 = 1056
with tf.variable_scope('conv1') as scope:
    images = tf.reshape(tensor=X,
                        shape=[-1, CHARACTER_HEIGHT, CHARACTER_WIDTH, 1])

    w1 = tf.get_variable(name='w1',
                         shape=[5, 5, 1, 16],
                         initializer=tf.truncated_normal_initializer())
    b1 = tf.get_variable(name='b1',
                         shape=[16],
                         initializer=tf.random_normal_initializer())

    conv1 = tf.nn.conv2d(input=images,
                         filter=w1,
                         strides=[1, 1, 1, 1],
                         padding='SAME',
                         name='conv1')
    conv1_relu = tf.nn.relu(features=conv1 + b1,
                            name='conv1_relu')

# Input 24 x 44 x 8 = 8448
with tf.variable_scope('pool1') as scope:
    pool1 = tf.nn.max_pool(value=conv1_relu,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool1')

# Input 8448 / 4 = 2112
with tf.variable_scope('conv2') as scope:
    w2 = tf.get_variable(name='w2',
                         shape=[5, 5, 16, 32],
                         initializer=tf.truncated_normal_initializer())
    b2 = tf.get_variable(name='b2',
                         shape=[32],
                         initializer=tf.random_normal_initializer())

    conv2 = tf.nn.conv2d(input=pool1,
                         filter=w2,
                         strides=[1, 1, 1, 1],
                         padding='SAME',
                         name='conv2')
    conv2_relu = tf.nn.relu(features=conv2 + b2,
                            name='conv2_relu')

# Input 2112 x 2 = 4224
with tf.variable_scope('pool2') as scope:
    pool2 = tf.nn.max_pool(value=conv2_relu,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool2')

# Input 4224 / 2 = 2112
with tf.variable_scope('fc') as scope:
    num_inputs = 2112
    w_fc = tf.get_variable(name='w_fc',
                           shape=[num_inputs, 256],
                           initializer=tf.truncated_normal_initializer())
    b_fc = tf.get_variable(name='b_fc',
                           shape=[256],
                           initializer=tf.random_normal_initializer())
    features = tf.reshape(pool2, [-1, num_inputs], name='features')

    fc = tf.nn.relu(features=tf.matmul(features, w_fc) + b_fc,
                    name='fc')

#
with tf.variable_scope('softmax') as scope:
    w_sm = tf.get_variable(name='w_sm',
                           shape=[256, NUM_CLASSES],
                           initializer=tf.truncated_normal_initializer())
    b_sm = tf.get_variable(name='b_sm',
                           shape=[NUM_CLASSES],
                           initializer=tf.random_normal_initializer())

    logits = tf.matmul(fc, w_sm) + b_sm
