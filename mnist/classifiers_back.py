'''
Slightly modified base on:
https://github.com/aymericdamien/TensorFlow-Examples/
'''
from __future__ import print_function

import tensorflow as tf
import utils
import os

flags = tf.app.flags

flags.DEFINE_string(
    'dataset', 'mnist', 'dataset from {mnist, fashion, svhn, cifar10}')

config = flags.FLAGS

class classifier(object):
    def __init__(self, sess):
        # Parameters
        self.sess = sess
        self.learning_rate = 0.0001
        self.epochs = 10
        self.batch_size = 64
        self.display_step = 10

        self.dataset = config.dataset

        # Network Parameters
        self.n_classes = 10 # mnist, svhn, cifar10 total classes
        self.dropout = 0.75 # Dropout, probability to keep units

        # tf Graph input
        if dataset in ['mnist', 'fashion']:
            self.x = tf.placeholder(tf.float32, [None, 28, 28, 1])
        elif dataset in ['svhn', 'cifar10']:
            self.x = tf.placeholder(tf.float32, [None, 32, 32, 3])

        self.y = tf.placeholder(tf.float32, [None, self.n_classes])
        self.keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

        # Create some wrappers for simplicity
        def conv2d(x, W, b, strides=1):
            # Conv2D wrapper, with bias and relu activation
            x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
            x = tf.nn.bias_add(x, b)
            return tf.nn.relu(x)

        def maxpool2d(x, k=2):
            # MaxPool2D wrapper
            return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                                  padding='SAME')

        # Create model
        def conv_net(x, weights, biases, dropout):
            # Convolution Layer
            conv1 = conv2d(x, weights['wc1'], biases['bc1'])
            # Max Pooling (down-sampling)
            conv1 = maxpool2d(conv1, k=2)

            # Convolution Layer
            conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
            # Max Pooling (down-sampling)
            conv2 = maxpool2d(conv2, k=2)

            # Fully connected layer
            # Reshape conv2 output to fit fully connected layer input
            fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
            fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
            fc1 = tf.nn.relu(fc1)
            # Apply Dropout
            fc1 = tf.nn.dropout(fc1, dropout)

            # Output, class prediction
            out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
            return out

        # Store layers weight & bias
        self.weights = {
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
            # 1024 inputs, 10 outputs (class prediction)
            'out': tf.Variable(tf.random_normal([1024, self.n_classes]))
        }

        self.biases = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

        # Construct model
        self.pred = conv_net(self.x, self.weights, self.biases, self.keep_prob)

        # Define loss and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # Evaluate model
        self.correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        self.saver = tf.train.Saver()

    def train(self, dataset):
        # Initializing the variables
        init = tf.global_variables_initializer()

        data, label, test_data, test_label = utils.load_mnist(dataset)
        # Launch the graph
        self.sess.run(init)
        step = 0
        # Keep training until reach max iterations
        while step < self.epochs:
            batch_idxs = len(data) // self.batch_size
            for idx in xrange(batch_idxs):
                batch_x = data[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_y = label[idx*self.batch_size:(idx+1)*self.batch_size]

                # Run optimization op (backprop)
                sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y,
                                               self.keep_prob: self.dropout})
                if idx % self.display_step == 0:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([self.cost, self.accuracy], feed_dict={self.x: batch_x,
                                                                      self.y: batch_y,
                                                                      self.keep_prob: 1.})
                    print("Epoch " + str(step) + " Iter " + str(idx*self.batch_size) + \
                          ", Minibatch Loss= " + "{:.6f}".format(loss) + \
                          ", Training Accuracy= " + "{:.5f}".format(acc))
            step += 1
            self.test(test_data[:1000], test_label[:1000])

        print("Optimization Finished!")

    def test(self, test_data, test_labels):
        # Calculate accuracy for 256 mnist test images
        print("Testing Accuracy:", \
            self.sess.run(self.accuracy, feed_dict={self.x: test_data,
                                          self.y: test_labels,
                                          self.keep_prob: 1.}))
    def predict(self, test_data):
        pred_prob = self.sess.run(tf.nn.softmax(self.pred), feed_dict={self.x: test_data,
                                                   self.keep_prob: 1.})
        return pred_prob

    def save(self, checkpoint_dir):
        model_name = "mnist_cnn_classifier"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=self.epochs)
    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints..")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(' [*] Success to read {}'.format(ckpt_name))
            return True, counter
        else:
            print(' [*] Failed to find a checkpoint')
            return False, 0

if __name__ == '__main__':
    dataset = 'mnist'
    with tf.Session() as sess:
        model = mnist_cnn(sess)
        model.train(dataset)
        model.save(dataset + '_cnn')
        model.load(dataset + '_cnn')
        data, label, test_data, test_label = utils.load_mnist(dataset)
        model.test(test_data[:5000], test_label[:5000])
