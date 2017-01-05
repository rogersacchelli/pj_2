import tensorflow as tf
import pickle
import os
import numpy as np
import random
import time
import cv2
import sys
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


# TENSORFLOW FLAGS
FLAGS = tf.app.flags.FLAGS

# IMAGE INFO FLAGS
tf.app.flags.DEFINE_integer('IMAGE_HEIGHT', '32', 'IMAGE HEIGHT SIZE')
tf.app.flags.DEFINE_integer('IMAGE_WIDTH', '32', 'IMAGE WIDTH SIZE')
tf.app.flags.DEFINE_integer('NUM_OF_CHAN', '3', 'IMAGE LAYERS')

# DATASET INFO
tf.app.flags.DEFINE_integer('NUM_OF_CLASSES', '43', 'NUMBER OF CLASSES')

# CNN PARAMETERS
tf.app.flags.DEFINE_float('start_learning_rate', '0.001', 'Start Learning Rate')
tf.app.flags.DEFINE_integer('batch_size', '256', 'Batch Size')
tf.app.flags.DEFINE_integer('epoch_size', '100', 'Epoch Size')

# FILE HANDLING FLAGS
tf.app.flags.DEFINE_string('check', 'checkpoint/leNet_for_traffic_signs_adagrad.ckpt', 'File name for model saving')

tf.app.flags.DEFINE_string('dataset_dir', 'traffic-signs-data', 'Train and test data set folder')
tf.app.flags.DEFINE_string('my_set_dir', 'traffic-signs-data/my_test_set', 'Personal data set folder')
tf.app.flags.DEFINE_string('train', 'train.p', 'train data set')
tf.app.flags.DEFINE_string('test', 'test.p', 'test data set')


layer_width = {
    'layer_1': 108,
    'layer_2': 108,
    'layer_3': 108,
    'fully_connected_1': 100,
    'fully_connected_2': 100,
    'fully_connected_3': 100
}

# Store layers weight & bias


def read_pickle(train=os.path.join(FLAGS.dataset_dir, FLAGS.train),
                test=os.path.join(FLAGS.dataset_dir, FLAGS.test)):
    # Unpickling data and load it on memory
    # Pickle file is structered as follows:
    # train = {'coords':[ndarray], 'features':[ndarray], 'labels':[ndarray], 'sizes':[ndarray]}
    # test = {'coords':[ndarray], 'features':[ndarray], 'labels':[ndarray], 'sizes':[ndarray]}

    print("\n--- READING DATASET FILES ---\n")

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
    print("\n---"
          "TRAIN DATASET STATS (To be splitted into Train and Validation ---")

    # Print number of examples per class
    samples_per_class = []
    for i in range(n_classes):
        samples_per_class.append(np.sum(train_dict['labels'][:] == i))
        print("\tClass %d: %d" % (i, samples_per_class[-1]))

    # Calculate Memory for whole set of training
    # Every feature will be mapped to float (4 Bytes)
    print("Required Memory for dataset: ",
          ((img_dim[0] * img_dim[1] * img_dim[2]) / 1024e2) * n_train * 4, 'MB')

    # Multiplot All Classes
    #plt.figure(figsize=(14, 10))
    #for i in range(n_classes):
    #    plt.subplot(5, 10, i + 1)
    #    plt.title('Class %d' % i)
    #    plt.imshow(train_dict['features'][sum(samples_per_class[0:i])])
    #plt.show()

    return train_dict, test_dict


def data_augmentation(X, y, X_valid, y_valid):
    # DATA AUGMENTATION RANDOMLY ADDS A SET OF IMAGE
    # TRANSFORMATION TO INCREASE THE NUMBER OF EXAMPLES PER CLASS
    # PROVIDING A IMPROVED DISTRIBUTION OF IMAGES PER CLASS

    print("\n--- DATA AUGMENTATION FOR TEST SET ---\n")

    def _affine(image):
        ang_range = 10
        trans_range = 10
        shear_range = 10

        ang_rot = np.random.uniform(ang_range) - ang_range / 2
        rows, cols, ch = image.shape
        Rot_M = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang_rot, 1)

        # Translation
        tr_x = trans_range * np.random.uniform() - trans_range / 2
        tr_y = trans_range * np.random.uniform() - trans_range / 2
        Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])

        # Shear
        pts1 = np.float32([[5, 5], [20, 5], [5, 20]])

        pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
        pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2

        pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])

        shear_M = cv2.getAffineTransform(pts1, pts2)

        image = cv2.warpAffine(image, Rot_M, (cols, rows))
        image = cv2.warpAffine(image, Trans_M, (cols, rows))
        image = cv2.warpAffine(image, shear_M, (cols, rows))

        return image

    def _median_blur(image):
        return cv2.medianBlur(image, 3)

    def _blur(image):
        return cv2.blur(image, (3, 3))

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


    samples_per_class = []

    # Loop through label set to check number of samples per class
    for i in range((np.max(y) + 1)):
        samples_per_class.append(np.sum(y == i))

    # AVG NUMBERS OF SAMPLES PER CLASS
    avg_sample_per_class = int(np.mean(samples_per_class))

    # STANDARD DEVIATION FOR DATASET - THE LOWER THE BEST
    # std_dev_dataset = int(np.std(samples_per_class))

    # NUMBER OF SAMPLES TO ADD PER CLASS
    samples_to_add_per_class = []
    for i, v in enumerate(samples_per_class):
        if v < avg_sample_per_class:
            samples_to_add_per_class.append(avg_sample_per_class - v)
        else:
            samples_to_add_per_class.append(0)

    # ADDING DATA PROCESSED

    print("AUGMENTING TRAINING DATA")
    while sum(samples_to_add_per_class) > 0:
        print("\t %d more samples to ADD" % sum(samples_to_add_per_class))
        for i in range(len(X)):
            if samples_to_add_per_class[y[i]] != 0:
                rand_func = random.randint(0, 2)
                if rand_func == 0:
                    new_feature = _median_blur(image=X[i])
                elif rand_func == 1:
                    new_feature = _histeq(image=X[i])
                elif rand_func == 2:
                    new_feature = _affine(image=X[i])
                    samples_to_add_per_class[y[i]] -= 1
                X = np.concatenate((X, np.expand_dims(new_feature, axis=0)))
                y = np.concatenate((y, np.expand_dims(y[i], axis=0)))

    # for i, v in enumerate(samples_to_add_per_class):
    #     print("ADJUSTING CLASS: %d ADDING %d SAMPLES" %(i,v))
    #     while v:
    #         start_pos = int(np.sum(samples_per_class[0:i]))
    #         pos = random.randint(0, samples_per_class[i])
    #         rand_func = random.randint(0, 2)
    #         if rand_func == 0:
    #             new_feature = _median_blur(image=X[start_pos + pos])
    #         elif rand_func == 1:
    #             new_feature = _histeq(image=X[start_pos + pos])
    #         elif rand_func == 2:
    #             new_feature = _affine(image=X[start_pos + pos])
    #
    #         X = np.concatenate((X, np.expand_dims(new_feature, axis=0)))
    #         y = np.concatenate((y, np.expand_dims(y[start_pos + pos], axis=0)))
    #
    #         v -= 1


    # CREATE NEW PICKLE FOR NEW AUGMENTED DATASET
    file = open('augmented_train.p', mode='wb')
    pickle.dump([X, y, X_valid, y_valid], file)

    return X, y

def read_stats(X, y):
    n_train = len(X)
    n_classes = np.max(y) + 1

    print("Number of training examples =", n_train)
    print("Number of classes =", n_classes)

    for i in range(n_classes):
        print('\tClass %d: %d' % (i, np.sum(y[:] == i)))

    print("Required Memory: ",
          ((32 * 32 * 3) / 1024e2) * n_train * 4, 'MB')

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

    # layer 1 - 32x32x3 to 28x28x108
    conv1 = conv_2d(x, w['layer_1'], b['layer_1'], s)

    # Local Response Normalization
    conv1 = lrn(conv=conv1)

    # Max Pooling -> 14x14x108
    conv1 = maxpool_2d(conv1)

    # layer 2 - 14x14x108 -> 10x10x108
    conv2 = conv_2d(conv1, w['layer_2'], b['layer_2'], s)

    # Local Response Normalization
    conv2 = lrn(conv=conv2)

    # Max Pooling -> 5x5x108
    conv2 = maxpool_2d(conv2)


    # layer 3 - 8x8x16 -> 8x8x36
    conv3 = conv_2d(conv2, w['layer_3'], b['layer_3'], s)
    # Max Pooling -> 4x4x46
    conv3 = maxpool_2d(conv3)


    # Fully connected layer 1 - 25*108
    fc1 = tf.reshape(conv3,[-1, w['fully_connected_1'].get_shape().as_list()[0]])
    # FC 1 2550 -> 100
    fc1 = tf.add(tf.matmul(fc1, w['fully_connected_1']), b['fully_connected_1'])
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

    my_features = np.zeros(shape=(len(image_list), 32, 32, 3), dtype=np.float64)
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

    return my_features, my_labels


def main():

    main_start_time = time.time()

    # READ FILES
    if not os.path.exists('./augmented_train.p'):
        # UNPICKLING DATA FROM FILES
        train_data, test_data = read_pickle()

        train_features = train_data['features']
        train_labels = train_data['labels']

        X_train, X_valid, y_train, y_valid = train_test_split(train_features, train_labels)

        print("\n--- TRAINING DATA ---")
        read_stats(X_train, y_train)

        ### DATA AUGMENTATION ###
        # ADD DISTORTION AND GENERATE MORE SAMPLES

        X_train, y_train = data_augmentation(X_train, y_train, X_valid, y_valid)

        print("\n--- AUGMENTED TRAINING DATA ---:")
        read_stats(X_train, y_train)

        X_train, y_train= shuffle(X_train, y_train)



    else:
       with open('augmented_train.p', mode='rb') as f:
           print("LOADING AUGMENTED TRAINING SET")
           X_train, y_train, X_valid, y_valid = pickle.load(f)
           _, test_data = read_pickle()
           read_stats(X_train, y_train)
           f.close()



    # ### PRE PROCESSING ###
    # SHUFFLING TRAINING DATA SET TO IMPROVE ACCURACY
    test_features, test_labels = shuffle(test_data['features'], test_data['labels'])

    # NORMALIZING DATA TO IMPROVE SGD CONVERGENCE
    X_train = normalize_images(X_train)
    test_features = normalize_images(test_features)

    x = tf.placeholder(tf.float32, shape=(None, FLAGS.IMAGE_HEIGHT, FLAGS.IMAGE_WIDTH, FLAGS.NUM_OF_CHAN))
    y = tf.placeholder(tf.int32, None)
    one_hot_y = tf.one_hot(y, FLAGS.NUM_OF_CLASSES)


    weights = {
        'layer_1': tf.Variable(tf.truncated_normal(
            [5, 5, 3, layer_width['layer_1']], stddev=0.001)),
        'layer_2': tf.Variable(tf.truncated_normal(
            [5, 5, layer_width['layer_1'], layer_width['layer_2']], stddev=0.001)),
        'layer_3': tf.Variable(tf.truncated_normal(
            [5, 5, layer_width['layer_2'], layer_width['layer_3']], stddev=0.001)),
        'fully_connected_1': tf.Variable(tf.truncated_normal(
            [1 * 1 * 108, layer_width['fully_connected_1']])),
        'fully_connected_2': tf.Variable(tf.truncated_normal(
            [layer_width['fully_connected_1'], layer_width['fully_connected_2']])),
        'out': tf.Variable(tf.truncated_normal(
            [layer_width['fully_connected_2'], FLAGS.NUM_OF_CLASSES]))
    }
    biases = {
        'layer_1': tf.Variable(tf.zeros(layer_width['layer_1'])),
        'layer_2': tf.Variable(tf.zeros(layer_width['layer_2'])),
        'layer_3': tf.Variable(tf.zeros(layer_width['layer_3'])),
        'fully_connected_1': tf.Variable(tf.zeros(layer_width['fully_connected_1'])),
        'fully_connected_2': tf.Variable(tf.zeros(layer_width['fully_connected_2'])),
        'out': tf.Variable(tf.zeros(FLAGS.NUM_OF_CLASSES))
    }

    logits = cnn(x, weights, biases)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y))
    # optimizer = tf.train.AdadeltaOptimizer(learning_rate=FLAGS.start_learning_rate).minimize(cost)
    optimizer = tf.train.AdagradOptimizer(learning_rate=FLAGS.start_learning_rate).minimize(cost)


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

    def _top_k_model_certainty(logits, features, labels, k=5):

        # RETURNS:
        # list:
        #   [0] Input features
        #   [1][0] = K TOP PREDICTIONS OF FEATURES (N, 32, 32, 3)
        #   [1][1] = K TOP LABELS OF FEATURES (N, 32, 32, 3)

        top_k = tf.nn.top_k(logits, k=5)
        predictions = sess.run([logits, top_k], feed_dict={x:normalize_images(features)})
        # Loop through predictions return
        for i in range(len(features)):
            plt.imshow(features[i])
            plt.show()
            print("------------------------------------")
            print("Sign %d - Correct Label %d" % (i, labels[i]) )
            print("Certainty: %s \t" % predictions[1][0][i])
            print("Labels: %s \t" % predictions[1][1][i])
            print("------------------------------------")



    with tf.Session() as sess:
        sess.run(init)

        total_batch = int(len(X_train) / FLAGS.batch_size)
        keep_prob = tf.placeholder(tf.float32)

        try:
            saver.restore(sess, os.path.join(os.curdir, FLAGS.check))
            print("Model Loaded")
        except:
            # If file does not exist, create dir not returning exception if dir exists
            os.makedirs(os.path.join(os.path.curdir, 'checkpoint'), exist_ok=True)

        for epoch in range(FLAGS.epoch_size):

            # Shuffling every epoch
            X_train, y_train = shuffle(X_train, y_train)

            # Loop over all batches
            for i in range(total_batch):

                start_time = time.time()
                start_batch_idx = i * FLAGS.batch_size
                end_batch_idx = start_batch_idx + FLAGS.batch_size
                batch_x = X_train[start_batch_idx: end_batch_idx]
                batch_y = y_train[start_batch_idx: end_batch_idx]

                if i == (total_batch - 1):
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
            if epoch % 10 == 0:
                # Save model state after each epoch
                saver.save(sess, os.path.join(os.getcwd(), FLAGS.check))
                print("Model Saved", FLAGS.check)
                print("Accuracy:", _evaluate(X_valid, y_valid))

        # EVALUATE TEST SET
        print("Test set accuracy:", _evaluate(test_features, test_labels))

        # EVALUATE PERSONAL SET
        my_feature_set, my_label_set = get_personal_data(FLAGS.my_set_dir)
        print("My test set accuracy:", _evaluate(normalize_images(my_feature_set), my_label_set))

        # CERTAINTY OF MODEL
        _top_k_model_certainty(logits, my_feature_set, my_label_set)

    print("Optimization Finished!")
    total_time = time.time() - main_start_time
    print("Total Time: ", total_time / 3600, " hours")

if __name__ == '__main__':
    main()
