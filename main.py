import tensorflow as tf
import pickle
import os
import numpy as np
import random
import time
import cv2
from sklearn.utils import shuffle

try:
    from sklearn.model_selection import train_test_split
except:
    from sklearn.cross_validation import train_test_split

from matplotlib import pyplot as plt

__autor__ = "Roger S. Sacchelli - roger.sacchelli@gmail.com"

__doc__ = """
    -----------------------------------------------
    ------- Project 2 | Self-driving Car ND -------
    ----- Classifying Traffic Signs with CNN ------
    -----------------------------------------------

    Model: Input -> CONV -> RELU -> LRN -> MAX_POOL -> CONV -> RELU -> LRN -> MAX_POOL -> FLAT -> FC -> FC -> OUT

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
tf.app.flags.DEFINE_float('start_learning_rate', '1.0', 'Start Learning Rate')
tf.app.flags.DEFINE_integer('batch_size', '256', 'Batch Size')
tf.app.flags.DEFINE_integer('epoch_size', '300', 'Epoch Size')

# FILE HANDLING FLAGS
tf.app.flags.DEFINE_string('check', 'checkpoint/leNet_for_traffic_signs_adadelta_test2.ckpt', 'File name for model saving')

tf.app.flags.DEFINE_string('dataset_dir', 'traffic-signs-data', 'Train and test data set folder')
tf.app.flags.DEFINE_string('my_set_dir', 'traffic-signs-data/my_test_set', 'Personal data set folder')
tf.app.flags.DEFINE_string('train', 'train.p', 'train data set')
tf.app.flags.DEFINE_string('test', 'test.p', 'test data set')


layer_width = {
    'layer_1': 6,
    'layer_2': 16,
    'fully_connected_1': 120,
    'fully_connected_2': 84
}

# Store layers weight & bias


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

    n_train = len(train_dict['features'])
    n_test = len(test_dict['features'])
    img_dim = train_dict['features'][0].shape
    n_classes = np.max(train_dict['labels'][:]) + 1

    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Image data shape =", img_dim)
    print("Number of classes =", n_classes)

    # Print number of examples per class
    samples_per_class = []
    for i in range(n_classes):
        samples_per_class.append(np.sum(train_dict['labels'][:] == i))
        print("\t Class", i, ": %d" % samples_per_class[-1], )

    # Calculate Memory for whole set of training
    # Every feature will be mapped to float (4 Bytes)
    print("Required Memory for validation whole data set: ",
          ((img_dim[0] * img_dim[1] * img_dim[2]) / 1024e2) * n_train * 4, 'MB')

    # Multiplot All Classes
    #plt.figure(figsize=(14, 10))
    #for i in range(n_classes):
    #    plt.subplot(5, 10, i + 1)
    #    plt.title('Class %d' % i)
    #    plt.imshow(train_dict['features'][sum(samples_per_class[0:i])])
    #plt.show()

    return train_dict, test_dict


def data_augmentation(dataset):
    # DATA AUGMENTATION RANDOMLY ADDS A SET OF IMAGE
    # TRANSFORMATION TO INCREASE THE NUMBER OF EXAMPLES PER CLASS
    # PROVIDING A IMPROVED DISTRIBUTION OF IMAGES PER CLASS

    def _flip_random_90_degrees(image, flip):
        # RANDOM FLIP
        """
        :param image: image np array
        :param flip: 90 degrees counter clock wise
        :return: rotated image
        """
        return np.rot90(image, flip)

    def _median_blur(image):
        return cv2.medianBlur(image, 3)

    def _blur(image):
        return cv2.blur(image, (3, 3))

    def _image_norm(image):
        return (np.subtract((image), 127.) / (max(np.std((image)),
              1. / (FLAGS.IMAGE_HEIGHT * FLAGS.IMAGE_WIDTH * FLAGS.NUM_OF_CHAN))))

    def _clahe(image):
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Method by Prateek Joshi - https://www.packtpub.com/mapt/book/Application-Development/9781785283932
        img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(img_lab)

        # convert the YUV image back to RGB format
        return cv2.cvtColor(cl1, cv2.COLOR_GRAY2BGR)

    def _histeq(image):

        # Method by Prateek Joshi - https://www.packtpub.com/mapt/book/Application-Development/9781785283932
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        # equalize the histogram of the Y channel
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

        # convert the YUV image back to RGB format
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        return img_output

    def _concat_new_data(dataset, new_feat, idx):
        dataset['features'] = np.concatenate((dataset['features'], np.expand_dims(new_feat, axis=0)))
        dataset['labels'] = np.concatenate((dataset['labels'], np.expand_dims(dataset['labels'][idx], axis=0)))
        dataset['coords'] = np.concatenate((dataset['coords'], np.expand_dims(dataset['coords'][idx], axis=0)))
        dataset['sizes'] = np.concatenate((dataset['sizes'], np.expand_dims(dataset['sizes'][idx], axis=0)))
        return dataset

    samples_per_class = []
    for i in range((np.max(dataset['labels'][:]) + 1)):
        samples_per_class.append(np.sum(dataset['labels'] == i))

    # AVG NUMBERS OF SAMPLES PER CLASS
    avg_sample_per_class = int(np.mean(samples_per_class))

    # STANDARD DEVIATION FOR DATASET - THE LOWER THE BEST
    std_dev_dataset = int(np.std(samples_per_class))

    # NUMBER OF SAMPLES TO ADD PER CLASS
    samples_to_add_per_class = []
    for i, v in enumerate(samples_per_class):
        if v < avg_sample_per_class:
            samples_to_add_per_class.append(std_dev_dataset)
        else:
            samples_to_add_per_class.append(0)

    # ADDING DATA
    for i, v in enumerate(samples_to_add_per_class):
        print("ADJUSTING CLASS:", i)
        while v:
            start_pos = int(np.sum(samples_per_class[0:i]))
            pos = random.randint(0, samples_per_class[i])
            rand_func = random.randint(0, 4)
            if rand_func == 0:
                new_feature = _flip_random_90_degrees(image=dataset['features'][start_pos + pos],
                                                      flip=random.randint(1, 3))
            elif rand_func == 1:
                new_feature = _median_blur(image=dataset['features'][start_pos + pos])
            elif rand_func == 2:
                new_feature = _blur(image=dataset['features'][start_pos + pos])
            elif rand_func == 3:
                new_feature = _histeq(image=dataset['features'][start_pos + pos])
            elif rand_func == 4:
                new_feature = _clahe(image=dataset['features'][start_pos + pos])

            dataset = _concat_new_data(dataset, new_feature, start_pos + pos)
            v -= 1

    # AUGMENTATED DATA STATS

    n_train = len(dataset['features'])
    img_dim = dataset['features'][0].shape
    n_classes = np.max(dataset['labels'][:]) + 1

    print("Number of training examples =", n_train)
    print("Image data shape =", img_dim)
    print("Number of classes =", n_classes)

    # Print number of examples per class
    samples_per_class = []
    for i in range(n_classes):
        samples_per_class.append(np.sum(dataset['labels'][:] == i))
        print("\t Class ", i, ": %d" % samples_per_class[-1], )

    # Calculate Memory for whole set of training
    # Every feature will be mapped to float (4 Bytes)
    print("Required Memory for validation whole dataset: ",
          ((img_dim[0] * img_dim[1] * img_dim[2]) / 1024e2) * n_train * 4, 'MB')

    # CREATE NEW PICKLE FOR NEW AUGMENTED DATASET
    file = open('augmented_train.p', mode='wb')
    pickle.dump(dataset, file)

    return dataset


def normalize_images(dataset):
    # THIS FUNCTION NORMALIZES IMAGES, COMPUTING:

    dataset = dataset.astype(dtype='float64')
    dataset -= np.mean(dataset, axis=0)
    dataset /= np.std(dataset, axis=0)

    return dataset


def sparse_to_dense(dataset):
    sp2dense = np.zeros(shape=(len(dataset['labels']), FLAGS.NUM_OF_CLASSES), dtype=int)
    for i in range(len(dataset['labels'])):
        sp2dense[i, dataset['labels'][i]] = 1

    dataset['labels'] = sp2dense

    return dataset


def shuffle_dataset(dataset):
    # SHUFFLE TRAIN DATA SET TO IMPROVE ACCURACY

    # SET OF FEATURES TO TEST RANDOM

    dataset['features'], dataset['labels'], dataset['coords'], dataset['sizes'] = \
        shuffle(dataset['features'], dataset['labels'], dataset['coords'], dataset['sizes'])

    plt.figure(figsize=(14, 10))
    for i in range(FLAGS.NUM_OF_CLASSES):
        array_class = np.where(dataset['labels'] == i)
        plt.subplot(5, 10, i + 1)
        plt.title('Class %d' % i)
        plt.imshow(dataset['features'][array_class[0][0]])
    plt.show()

    return dataset


def conv_2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    #return tf.nn.tanh(x)
    return tf.nn.relu(x)


def maxpool_2d(x, k=2):
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='SAME')


def lrn(conv, radius=2, alpha=2e-05, beta=0.75, bias=1.0):
    return tf.nn.local_response_normalization(conv, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)


# MODEL FOR CNN
def cnn(x, w, b, s=1, dropout=0.5):
    # CONVOLUTIONAL NEURAL NET

    """:param: input, weights, biases and strides"""

    # layer 1 - 32x32x3 to 32x32x6
    conv1 = conv_2d(x, w['layer_1'], b['layer_1'], s)

    # Local Response Normalization
    conv1 = lrn(conv=conv1)

    # Max Pooling -> 16x16x6
    conv1 = maxpool_2d(conv1)

    # layer 2 - 16x16x6 -> 16x16x16
    conv2 = conv_2d(conv1, w['layer_2'], b['layer_2'], s)

    # Local Response Normalization
    conv2 = lrn(conv=conv2)

    # Max Pooling -> 8x8x16
    conv2 = maxpool_2d(conv2)

    # layer 3 - 8x8x16 -> 8x8x36
    #conv3 = conv_2d(conv2, w['layer_3'], b['layer_3'], s)
    # Max Pooling -> 4x4x46
    #conv3 = maxpool_2d(conv3)

    # Fully connected layer 1 - 4*4*36 to 576
    fc1 = tf.reshape(
        conv2,
        [-1, w['fully_connected_1'].get_shape().as_list()[0]])
    # FC 1 4000 -> 120
    fc1 = tf.add(
        tf.matmul(fc1, w['fully_connected_1']),
        b['fully_connected_1'])
    fc1 = tf.nn.relu(fc1)

    # Dropout regularization for FC 1
    drop_fc1 = tf.nn.dropout(fc1, dropout)

    # Fully connected layer 2 - 120 to 120
    fc2 = tf.add(
        tf.matmul(drop_fc1, w['fully_connected_2']),
        b['fully_connected_2'])
    fc2 = tf.nn.relu(fc2)

    # Output Layer - class prediction - 512 to 43
    out = tf.add(tf.matmul(fc2, w['out']), b['out'])
    return out


def get_personal_data(path_to_dataset):
    image_list = os.listdir(os.path.join(os.getcwd(), path_to_dataset))

    import scipy.misc

    my_features = np.zeros(shape=(len(image_list), 32, 32, 3), dtype=np.float32)
    my_labels = np.zeros(shape=(len(image_list)), dtype=np.int8)

    plt.figure(figsize=(14, 10))
    for i, v in enumerate(image_list):
        img_path = os.path.join(os.getcwd(), FLAGS.my_set_dir, v)
        img = scipy.misc.imread(img_path)
        im_resized = scipy.misc.imresize(img, size=(32, 32))
        im_class = v.split('_')[0]
        my_features[i, :, :] = im_resized
        my_labels[i] = int(im_class)

        plt.subplot(5, 4, i + 1)
        plt.title('Class %s' % im_class)
        plt.imshow(im_resized)

    plt.show()

    return normalize_images(my_features), my_labels


def main():

    main_start_time = time.time()

    # READ FILES
    if not os.path.exists('./augmented_train.p'):
        # UNPICKLING DATA FROM FILES
        train_data, test_data = read_pickle()

        # ### DATA AUGMENTATION ###
        # ADD DISTORTION AND GENERATE MORE SAMPLES
        train_data = data_augmentation(train_data)
    else:
        with open('augmented_train.p', mode='rb') as f:
            print("LOADING AUGMENTED TRAINING SET")
            train_data = pickle.load(f)
            _, test_data = read_pickle()
            f.close()

    # ### PRE PROCESSING ###
    # SHUFFLING TRAINING DATA SET TO IMPROVE ACCURACY
    train_data = shuffle_dataset(train_data)
    test_data = shuffle_dataset(test_data)

    # NORMALIZING DATA TO IMPROVE SGD CONVERGENCE
    train_data_features = normalize_images(train_data['features'])
    test_data_features = normalize_images(test_data['features'])

    # TRANSFORM SPARSE LABELS MATRIX TO DENSE MATRIX
    # train_data = sparse_to_dense(train_data)
    # test_data = sparse_to_dense(test_data)

    X_train, X_valid, y_train, y_valid = train_test_split(train_data_features,
                                                          train_data['labels'], test_size=0.3)

    x = tf.placeholder(tf.float32, shape=(None, FLAGS.IMAGE_HEIGHT, FLAGS.IMAGE_WIDTH, FLAGS.NUM_OF_CHAN))
    y = tf.placeholder(tf.int32, None)
    one_hot_y = tf.one_hot(y, FLAGS.NUM_OF_CLASSES)

    weights = {
        'layer_1': tf.Variable(tf.truncated_normal(
            [5, 5, 3, layer_width['layer_1']], stddev=0.01)),
        'layer_2': tf.Variable(tf.truncated_normal(
            [5, 5, layer_width['layer_1'], layer_width['layer_2']], stddev=0.01)),
        #'layer_3': tf.Variable(tf.truncated_normal(
        #    [5, 5, layer_width['layer_2'], layer_width['layer_3']], stddev=0.01)),
        'fully_connected_1': tf.Variable(tf.truncated_normal(
            [400, layer_width['fully_connected_1']])),
        'fully_connected_2': tf.Variable(tf.truncated_normal(
            [layer_width['fully_connected_1'], layer_width['fully_connected_2']])),
        'out': tf.Variable(tf.truncated_normal(
            [layer_width['fully_connected_2'], FLAGS.NUM_OF_CLASSES]))
    }
    biases = {
        'layer_1': tf.Variable(tf.zeros(layer_width['layer_1'])),
        'layer_2': tf.Variable(tf.zeros(layer_width['layer_2'])),
        #'layer_3': tf.Variable(tf.zeros(layer_width['layer_3'])),
        'fully_connected_1': tf.Variable(tf.zeros(layer_width['fully_connected_1'])),
        'fully_connected_2': tf.Variable(tf.zeros(layer_width['fully_connected_2'])),
        'out': tf.Variable(tf.zeros(FLAGS.NUM_OF_CLASSES))
    }

    logits = cnn(x, weights, biases)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y))
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=FLAGS.start_learning_rate).minimize(cost)

    # Create saving object

    saver = tf.train.Saver()

    # Initializing the variables
    # Handle exception in case of tensorflow different versions
    try:
        # 0.12 Method
        init = tf.global_variables_initializer()
    except:
        # 0.11 Method
        init = tf.initialize_all_variables()

    def _accuracy(predictions, labels):
        return (np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
                / predictions.shape[0])

    def _evaluate(X_data, y_data):

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        num_examples = len(X_data)

        total_accuracy = 0
        session = tf.get_default_session()
        for offset in range(0, num_examples, FLAGS.batch_size):
            bat_x, bat_y = X_data[offset:offset + FLAGS.batch_size], \
                               y_data[offset:offset + FLAGS.batch_size]
            accuracy = session.run(accuracy_operation, feed_dict={x: bat_x, y: bat_y})
            total_accuracy += (accuracy * len(bat_x))
        return total_accuracy / num_examples

    with tf.Session() as sess:
        sess.run(init)

        total_batch = int(len(X_train) / FLAGS.batch_size)
        keep_prob = tf.placeholder(tf.float32)

        if os.path.exists(os.path.join(os.curdir, FLAGS.check)):
            saver.restore(sess, os.path.join(os.curdir, FLAGS.check))
            print("Model Loaded")
        else:
            # If file does not exist, create dir not returning exception if dir exists
            os.makedirs(os.path.join(os.path.curdir, 'checkpoint'), exist_ok=True)

        for epoch in range(FLAGS.epoch_size):

            # Loop over all batches
            for i in range(total_batch):

                start_time = time.time()
                start_batch_idx = i * FLAGS.batch_size
                end_batch_idx = start_batch_idx + FLAGS.batch_size
                batch_x = X_train[start_batch_idx: end_batch_idx]
                batch_y = y_train[start_batch_idx: end_batch_idx]

                if i % (total_batch / 2) == 1 or i == (total_batch - 1):
                    _, c = sess.run([optimizer, cost],
                                    feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
                    # Display logs per epoch step
                    batch_time = time.time() - start_time
                    print("Epoch:", '[%d' % (epoch + 1), 'of %d]' % FLAGS.epoch_size,
                          "| batch: [%d" % i, "of %d]" % total_batch,
                          "| cost =", "{:.3f}".format(c),
                          "| batch time: %.03f" % batch_time,
                          "| img/sec: %d" % int(FLAGS.batch_size / batch_time))
                else:
                    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

            # Save model state after each epoch
            saver.save(sess, os.path.join(os.getcwd(), FLAGS.check))
            print("Model Saved", FLAGS.check)
            print("Accuracy:", _evaluate(X_valid, y_valid))

        # EVALUATE TEST SET
        print("Test set accuracy:", _evaluate(test_data_features, test_data['labels']))

        # EVALUATE PERSONAL SET
        my_feature_set, my_label_set = get_personal_data(FLAGS.my_set_dir)
        print("My test set accuracy:", _evaluate(my_feature_set, my_label_set))

    print("Optimization Finished!")
    total_time = time.time() - main_start_time
    print("Total Time: ", total_time / 3600, " hours")

if __name__ == '__main__':
    main()
