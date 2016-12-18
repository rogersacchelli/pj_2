import tensorflow as tf
import pickle
import os
import numpy as np
import random
import argparse as argp

__autor__ = "Roger S. Sacchelli - roger.sacchelli@gmail.com"

__doc__ = """
    -----------------------------------------------
    ------- Project 2 | Self-driving Car ND -------
    ----- Classifying Traffic Signs with CNN ------
    -----------------------------------------------
    """


# FLAGS
FLAGS = tf.app.flags.FLAGS

# IMAGE INFO FLAGS
tf.app.flags.DEFINE_integer('IMAGE_HEIGHT', '32', 'IMAGE HEIGHT SIZE')
tf.app.flags.DEFINE_integer('IMAGE_WIDTH', '32', 'IMAGE WIDTH SIZE')
tf.app.flags.DEFINE_integer('NUM_OF_CHAN', '3', 'IMAGE LAYERS')

# DATASET INFO
tf.app.flags.DEFINE_integer('NUM_OF_CLASSES', '43', 'NUMBER OF CLASSES')

# CNN PARAMETERS
tf.app.flags.DEFINE_float('learning_rate', '0.001', 'Learning Rate')
tf.app.flags.DEFINE_integer('batch_size', '128', 'Batch Size')
tf.app.flags.DEFINE_integer('epoch_size', '30', 'Epoch Size')

# FILE HANDLING FLAGS
tf.app.flags.DEFINE_string('check', 'checkpoint/model_1.ckpt', 'File name for model saving')

tf.app.flags.DEFINE_string('dataset_dir', 'traffic-signs-data', 'Train and test dataset folder')
tf.app.flags.DEFINE_string('train', 'train.p', 'train dataset')
tf.app.flags.DEFINE_string('test', 'test.p', 'test dataset')



layer_width = {
    'layer_1': 32,
    'layer_2': 64,
    'layer_3': 128,
    'fully_connected': 512
}

# Store layers weight & bias
weights = {
    'layer_1': tf.Variable(tf.truncated_normal(
        [5, 5, 3, layer_width['layer_1']], stddev=0.1)),
    'layer_2': tf.Variable(tf.truncated_normal(
        [5, 5, layer_width['layer_1'], layer_width['layer_2']], stddev=0.1)),
    'layer_3': tf.Variable(tf.truncated_normal(
        [5, 5, layer_width['layer_2'], layer_width['layer_3']], stddev=0.1)),
    'fully_connected': tf.Variable(tf.truncated_normal(
        [4*4*128, layer_width['fully_connected']])),
    'out': tf.Variable(tf.truncated_normal(
        [layer_width['fully_connected'], FLAGS.NUM_OF_CLASSES]))
}
biases = {
    'layer_1': tf.Variable(tf.zeros(layer_width['layer_1'])),
    'layer_2': tf.Variable(tf.zeros(layer_width['layer_2'])),
    'layer_3': tf.Variable(tf.zeros(layer_width['layer_3'])),
    'fully_connected': tf.Variable(tf.zeros(layer_width['fully_connected'])),
    'out': tf.Variable(tf.zeros(FLAGS.NUM_OF_CLASSES))
}


def read_pickle(train=os.path.join(FLAGS.dataset_dir, FLAGS.train),
                test=os.path.join(FLAGS.dataset_dir, FLAGS.test)):
    # Unpickling data and load it on memory
    # Pickle file is structered as follows:
    # train = {'coords':[ndarray], 'features':[ndarray], 'labels':[ndarray], 'sizes':[ndarray]}
    # test = {'coords':[ndarray], 'features':[ndarray], 'labels':[ndarray], 'sizes':[ndarray]}

    with open(train, mode='rb') as f:
        train_dict = pickle.load(f)

    with open(test, mode='rb') as f:
        test_dict = pickle.load(f)

    return train_dict, test_dict


def normalize_images(train, test):
    # THIS FUNCTION NORMALIZES IMAGES, COMPUTING:
    # (X - MEAN)/ADJUSTED_STD_DEV WHERE ADJUSTED_STD_DEV = max(ADJUSTED_STD_DEV, 1.0/sqrt(image.NumElements()))

    # DATA TYPE CONVERSION
    train['features'] = train['features'].astype(np.float)
    test['features'] = test['features'].astype(np.float)

    # NORMALIZATION
    for i in range(len(train['features'][:])):
        train['features'][i] = (np.subtract((train['features'][i]), 127.) / (max(np.std((train['features'][i])),
                                                                                 1. / (
                                                                                 FLAGS.IMAGE_HEIGHT * FLAGS.IMAGE_WIDTH * FLAGS.NUM_OF_CHAN))))

    for i in range(len(test['features'][:])):
        test['features'][i] = (np.subtract((test['features'][i]), 127.) / (max(np.std((test['features'][i])),
                                                                               1. / (
                                                                               FLAGS.IMAGE_HEIGHT * FLAGS.IMAGE_WIDTH * FLAGS.NUM_OF_CHAN))))

    return train, test


def sparse_to_dense(dataset):
    sp2dense = np.zeros(shape=(len(dataset['labels']), FLAGS.NUM_OF_CLASSES))
    for i in range(len(dataset['labels'])):
        sp2dense[i, dataset['labels'][i]] = 1

    dataset['labels'] = sp2dense

    return dataset


def shuffle_dataset(dataset):
    # SHUFFLE TRAIN DATA SET TO IMPROVE ACCURACY

    for i in range(len(dataset['features'])):
        # RANDOM INTEGER
        rand_int = random.randint(0, len(dataset['features']) - 1)

        # RANDOM POSITION FROM DATASET
        temp = [dataset['coords'][rand_int], dataset['features'][rand_int],
                dataset['labels'][rand_int], dataset['sizes'][rand_int]]

        # ASSIGN RANDOM POSITION TO CURRENT POSITION
        dataset['coords'][rand_int] = dataset['coords'][i]
        dataset['features'][rand_int] = dataset['features'][i]
        dataset['labels'][rand_int] = dataset['labels'][i]
        dataset['sizes'][rand_int] = dataset['sizes'][i]

        # REPLACE CURRENT POSITION FROM RANDOM POSITION
        dataset['coords'][i] = temp[0]
        dataset['features'][i] = temp[1]
        dataset['labels'][i] = temp[2]
        dataset['sizes'][i] = temp[3]

    return dataset


def conv_2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.tanh(x)


def maxpool_2d(x, k=2):
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='SAME')


# MODEL FOR CNN
def cnn(x, w, b, s=1):
    # CONVOLUTIONAL NEURAL NET

    """:param: input, weights, biases and strides"""

    # layer 1 - 32x32x3 to 16x16x32
    conv1 = conv_2d(x, w['layer_1'], b['layer_1'], s)
    conv1 = maxpool_2d(conv1)

    # layer 2 - 16x16x32 - 8x8x64
    conv2 = conv_2d(conv1, w['layer_2'], b['layer_2'], s)
    conv2 = maxpool_2d(conv2)

    # layer 3 - 8x8x64 - 4x4x128
    conv3 = conv_2d(conv2, w['layer_3'], b['layer_3'], s)
    conv3 = maxpool_2d(conv3)

    # Fully connected layer - 4*4*128 to 512
    # Reshape conv3 output to fit fully connected layer input
    fc1 = tf.reshape(
        conv3,
        [-1, w['fully_connected'].get_shape().as_list()[0]])
    fc1 = tf.add(
        tf.matmul(fc1, w['fully_connected']),
        b['fully_connected'])
    fc1 = tf.nn.tanh(fc1)

    # Output Layer - class prediction - 512 to 32
    out = tf.add(tf.matmul(fc1, w['out']), b['out'])
    return out


def main():

    # UNPICKLING DATA FROM FILES
    train_data, test_data = read_pickle()

    # NORMALIZING DATA TO IMPROVE SGD CONVERGENCE
    train_data, test_data = normalize_images(train_data, test_data)

    # TRANSFORM SPARSE LABELS MATRIX TO DENSE MATRIX
    train_data = sparse_to_dense(train_data)
    test_data = sparse_to_dense(test_data)

    # SHUFFLING TRAINING DATA SET TO IMPROVE ACCURACY
    train_data = shuffle_dataset(train_data)
    test_data = shuffle_dataset(test_data)

    graph = tf.Graph()

    with graph.as_default():
        input = tf.placeholder(tf.float32, shape=(None, FLAGS.IMAGE_HEIGHT, FLAGS.IMAGE_WIDTH, FLAGS.NUM_OF_CHAN))
        labels = tf.placeholder(tf.float32, shape=(None, FLAGS.NUM_OF_CLASSES))

        weights = {
            'layer_1': tf.Variable(tf.truncated_normal(
                [5, 5, 3, layer_width['layer_1']], stddev=0.1)),
            'layer_2': tf.Variable(tf.truncated_normal(
                [5, 5, layer_width['layer_1'], layer_width['layer_2']], stddev=0.1)),
            'layer_3': tf.Variable(tf.truncated_normal(
                [5, 5, layer_width['layer_2'], layer_width['layer_3']], stddev=0.1)),
            'fully_connected': tf.Variable(tf.truncated_normal(
                [4 * 4 * 128, layer_width['fully_connected']])),
            'out': tf.Variable(tf.truncated_normal(
                [layer_width['fully_connected'], FLAGS.NUM_OF_CLASSES]))
        }
        biases = {
            'layer_1': tf.Variable(tf.zeros(layer_width['layer_1'])),
            'layer_2': tf.Variable(tf.zeros(layer_width['layer_2'])),
            'layer_3': tf.Variable(tf.zeros(layer_width['layer_3'])),
            'fully_connected': tf.Variable(tf.zeros(layer_width['fully_connected'])),
            'out': tf.Variable(tf.zeros(FLAGS.NUM_OF_CLASSES))
        }

        logits = cnn(input, weights, biases)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate) \
            .minimize(cost)

        # Create saving object

        saver = tf.train.Saver()

        # Initializing the variables
        init = tf.initialize_all_variables()

    # Launch the graph
        with tf.Session() as sess:
            sess.run(init)

            # Check for saved models
            if os.path.exists(os.path.join(os.curdir, FLAGS.check)):
                saver.restore(sess, os.path.join(os.curdir, FLAGS.check))
                print("Model Loaded")
            else:
                # If file does not exist, create dir not returning exception if dir exists
                os.makedirs(os.path.join(os.path.curdir, 'checkpoint'), exist_ok=True)

            # Training cycle
            offset = 0
            for epoch in range(FLAGS.epoch_size):
                total_batch = int(len(train_data['features']) / FLAGS.learning_rate)
                # Loop over all batches
                for i in range(total_batch):
                    batch_x = train_data['features'][offset:(i * FLAGS.batch_size)]
                    batch_y = train_data['labels'][offset:(i * FLAGS.batch_size)]
                    # Run optimization op (backprop) and cost op (to get loss value)
                    sess.run(optimizer, feed_dict={input: batch_x, labels: batch_y})
                    if i % 10 == 0:
                        # Display logs per epoch step
                        c = sess.run(cost, feed_dict={input: batch_x, labels: batch_y})
                        print("Epoch:", '%04d' % (epoch + 1), "batch:", i, "cost =", "{:.9f}".format(c))
                        # Save model state
                        saver.save(sess, os.path.join(os.curdir, FLAGS.check))
                        print("Model Saved")
                # Test model
                correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
                # Calculate accuracy
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                print(
                    "Accuracy:",
                    accuracy.eval({input: test_data['features'][:], labels: test_data['labels'][:]}))
            print("Optimization Finished!")




if __name__ == '__main__':
    main()
