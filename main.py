import tensorflow as tf
import pickle
import os

# FLAGS
FLAGS = tf.app.flags.FLAGS

# FILE HANDLING FLAGS
tf.app.flags.DEFINE_string('dataset_dir', 'traffic-signs-data', 'Train and test dataset folder')
tf.app.flags.DEFINE_string('train', 'train.p', 'train dataset')
tf.app.flags.DEFINE_string('test', 'test.p', 'test dataset')

# IMAGE INFO FLAGS
tf.app.flags.DEFINE_integer('IMAGE_HEIGHT', '32', 'IMAGE HEIGHT SIZE')
tf.app.flags.DEFINE_integer('IMAGE_WIDTH', '32', 'IMAGE HEIGHT SIZE')


def read_input(train=os.path.join(FLAGS.dataset_dir, FLAGS.train),
               test=os.path.join(FLAGS.dataset_dir, FLAGS.test)):
    # Unpickling data and load it on memory
    with open(train, mode='rb') as f:
        train_dict = pickle.load(f)

    with open(test, mode='rb') as f:
        test_dict = pickle.load(f)

    return train_dict, test_dict


def main():
    train_data = read_input()

    with tf.Session() as sess:
        tf.initialize_all_variables()
        sess.run()


if __name__ == '__main__':
    main()
