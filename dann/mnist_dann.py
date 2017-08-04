import tensorflow as tf
import numpy as np
import cPickle as pkl
from sklearn.manifold import TSNE

from flip_gradient import flip_gradient
from utils import *

from tensorflow.examples.tutorials.mnist import input_data

import argparse

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--model_mode', type=str, default='gan',
            help='mode from {gan, dan_s, dan_2s}')
    args = arg_parser.parse_args()
    model_mode = args.model_mode
    lbd = 0.2

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    batch_size = 128

    # Process MNIST
    mnist_train = (mnist.train.images > 0).reshape(55000, 28, 28, 1).astype(np.uint8) * 255
    mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
    mnist_test = (mnist.test.images > 0).reshape(10000, 28, 28, 1).astype(np.uint8) * 255
    mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)

    # Load MNIST-M
    mnistm = pkl.load(open('mnistm_data.pkl'))
    mnistm_train = mnistm['train']
    mnistm_test = mnistm['test']
    mnistm_valid = mnistm['valid']

    # Compute pixel mean for normalizing data
    pixel_mean = np.vstack([mnist_train, mnistm_train]).mean((0, 1, 2))

    class MNISTModel(object):
        """Simple MNIST domain adaptation model."""
        def __init__(self):
            self._build_model()

        def _build_model(self):

            self.X = tf.placeholder(tf.uint8, [None, 28, 28, 3])
            self.y = tf.placeholder(tf.float32, [None, 10])
            self.domain = tf.placeholder(tf.float32, [None, 2])

            self.domain_set = None
            if model_mode.endswith( '_s' ):
                self.domain_set = tf.placeholder(tf.float32, [2, 2])
            elif model_mode.endswith( '_2s' ):
                self.domain_set = tf.placeholder(tf.float32, [4, 2])

            self.l = tf.placeholder(tf.float32, [])
            self.train = tf.placeholder(tf.bool, [])

            X_input = (tf.cast(self.X, tf.float32) - pixel_mean) / 255.

            # CNN model for feature extraction
            with tf.variable_scope('feature_extractor'):

                W_conv0 = weight_variable([5, 5, 3, 32])
                b_conv0 = bias_variable([32])
                h_conv0 = tf.nn.relu(conv2d(X_input, W_conv0) + b_conv0)
                h_pool0 = max_pool_2x2(h_conv0)

                W_conv1 = weight_variable([5, 5, 32, 48])
                b_conv1 = bias_variable([48])
                h_conv1 = tf.nn.relu(conv2d(h_pool0, W_conv1) + b_conv1)
                h_pool1 = max_pool_2x2(h_conv1)

                # The domain-invariant feature
                self.feature = tf.reshape(h_pool1, [-1, 7*7*48])

            # MLP for class prediction
            with tf.variable_scope('label_predictor'):

                # Switches to route target examples (second half of batch) differently
                # depending on train or test mode.
                all_features = lambda: self.feature
                source_features = lambda: tf.slice(self.feature, [0, 0], [batch_size / 2, -1])
                classify_feats = tf.cond(self.train, source_features, all_features)

                all_labels = lambda: self.y
                source_labels = lambda: tf.slice(self.y, [0, 0], [batch_size / 2, -1])
                self.classify_labels = tf.cond(self.train, source_labels, all_labels)

                W_fc0 = weight_variable([7 * 7 * 48, 100])
                b_fc0 = bias_variable([100])
                h_fc0 = tf.nn.relu(tf.matmul(classify_feats, W_fc0) + b_fc0)

                W_fc1 = weight_variable([100, 100])
                b_fc1 = bias_variable([100])
                h_fc1 = tf.nn.relu(tf.matmul(h_fc0, W_fc1) + b_fc1)

                W_fc2 = weight_variable([100, 10])
                b_fc2 = bias_variable([10])
                logits = tf.matmul(h_fc1, W_fc2) + b_fc2

                self.pred = tf.nn.softmax(logits)
                self.pred_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.classify_labels, logits=logits )

            # Small MLP for domain prediction with adversarial loss
            with tf.variable_scope('domain_predictor'):

                # Flip the gradient when backpropagating through this operation
                feat = flip_gradient(self.feature, self.l)

                d_W_fc0 = weight_variable([7 * 7 * 48, 100])
                d_b_fc0 = bias_variable([100])
                d_h_fc0 = tf.nn.relu(tf.matmul(feat, d_W_fc0) + d_b_fc0)

                d_W_fc1 = weight_variable([100, 2])
                d_b_fc1 = bias_variable([2])
                d_logits = tf.matmul(d_h_fc0, d_W_fc1) + d_b_fc1

                self.domain_pred = tf.nn.softmax(d_logits)
                self.domain_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.domain, logits=d_logits)
            if 'dan' in model_mode:
                with tf.variable_scope('domain_set_predictor'):
                    # Flip the gradient when backpropagating through this operation
                    feat = flip_gradient(self.feature, self.l)

                    s_W_fc0 = weight_variable([7 * 7 * 48, 100])
                    s_b_fc0 = bias_variable([100])

                    s_W_fc1 = weight_variable([100, 2])
                    s_b_fc1 = bias_variable([2])

                    if model_mode.endswith( '_s' ):
                        avg_0 = tf.reduce_mean(feat[:int(batch_size/2)], axis=0, keep_dims=True)
                        avg_1 = tf.reduce_mean(feat[int(batch_size/2):], axis=0, keep_dims=True)
                        avg = tf.concat([avg_0, avg_1], 0)
                    elif model_mode.endswith( '_2s' ):
                        avgs = []
                        for i in range(4):
                            avgs.append(tf.reduce_mean(
                                feat[int(i*batch_size/4):int((i+1)*batch_size/4)],
                                axis=0, keep_dims=True))
                        avg_00 = tf.abs(avgs[0] - avgs[1])
                        avg_11 = tf.abs(avgs[2] - avgs[3])
                        avg_01 = tf.abs(avgs[0] - avgs[2])
                        avg_10 = tf.abs(avgs[1] - avgs[3])
                        avg = tf.concat([avg_00, avg_11, avg_01, avg_10], 0)

                    s_h_fc0 = tf.nn.relu(tf.matmul(avg, s_W_fc0) + s_b_fc0)
                    self.s_logits = tf.matmul(s_h_fc0, s_W_fc1) + s_b_fc1
                    self.s_softmax_logits = tf.nn.softmax(self.s_logits)
                    self.set_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.domain_set, logits=self.s_logits)

    # Build the model graph
    graph = tf.get_default_graph()
    with graph.as_default():
        model = MNISTModel()

        learning_rate = tf.placeholder(tf.float32, [])

        pred_loss = tf.reduce_mean(model.pred_loss)
        dann_loss = tf.reduce_mean(model.domain_loss)
        dan_loss = 0
        if 'dan' in model_mode:
            dan_loss = lbd * tf.reduce_mean(model.set_loss)

        total_loss = pred_loss + dann_loss + dan_loss

        regular_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(pred_loss)
        total_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_loss)

        # Evaluation
        correct_label_pred = tf.equal(tf.argmax(model.classify_labels, 1), tf.argmax(model.pred, 1))
        label_acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))
        correct_domain_pred = tf.equal(tf.argmax(model.domain, 1), tf.argmax(model.domain_pred, 1))
        domain_acc = tf.reduce_mean(tf.cast(correct_domain_pred, tf.float32))

    # Params
    num_steps = 55000

    def train_and_evaluate(graph, model, verbose=True):
        """Helper to run the model with different training modes."""

        with tf.Session(graph=graph) as sess:
            tf.global_variables_initializer().run()

            # Batch generators
            gen_source_batch = batch_generator(
                [mnist_train, mnist.train.labels], batch_size / 2)
            gen_target_batch = batch_generator(
                [mnistm_train, mnist.train.labels], batch_size / 2)

            domain_labels = np.vstack([np.tile([1., 0.], [batch_size / 2, 1]),
                                       np.tile([0., 1.], [batch_size / 2, 1])])
            domain_set_labels = None
            if model_mode.endswith( '_s' ):
                domain_set_labels = np.array([[1., 0.], [0., 1.]])
            elif model_mode.endswith( '_2s' ):
                domain_set_labels = np.array([[1., 0.], [1., 0.],
                                              [0., 1.], [0., 1.]])

            # Training loop
            i = 0
            while i < num_steps:
            #for i in range(num_steps):
                # Adaptation param and learning rate schedule as described in the paper
                p = float(i) / num_steps
                l = 2. / (1. + np.exp(-10. * p)) - 1
                lr = 0.01 / (1. + 10 * p)**0.75

                # Training step
                X0, y0 = gen_source_batch.next()
                X1, y1 = gen_target_batch.next()
                X = np.vstack([X0, X1])
                y = np.vstack([y0, y1])

                danloss = 0

                if 'dan' in model_mode:
                    _, loss, dloss, danloss, ploss, dacc, pacc = \
                            sess.run([total_train_op, \
                                      total_loss, dann_loss, dan_loss, pred_loss, \
                                      domain_acc, label_acc],
                                    feed_dict={model.X: X, model.y: y, \
                                               model.domain: domain_labels, \
                                               model.domain_set: domain_set_labels, \
                                               model.l: l, learning_rate: lr, \
                                               model.train: True})
                else:
                    _, loss, dloss, ploss, dacc, pacc = \
                            sess.run([total_train_op, \
                                      total_loss, dann_loss, pred_loss, \
                                      domain_acc, label_acc],
                                    feed_dict={model.X: X, model.y: y, \
                                               model.domain: domain_labels, \
                                               model.l: l, learning_rate: lr, \
                                               model.train: True})

                if verbose and i % 1000 == 0:
                    print('total_loss: {}, dann_loss: {}, dan_loss: {}, domain_acc: {}, class_acc: {}'.format(\
                            loss, dloss, danloss, dacc, pacc))

                i += 1
                if i > 1000 and pacc < 0.3:
                    print('restart!')
                    i = 0
                    tf.global_variables_initializer().run()

            # Compute final evaluation on test data
            source_acc = sess.run(label_acc,
                                feed_dict={model.X: mnist_test, model.y: mnist.test.labels,
                                           model.train: False})

            target_acc = sess.run(label_acc,
                                feed_dict={model.X: mnistm_test, model.y: mnist.test.labels,
                                           model.train: False})

        return source_acc, target_acc


    print '\nTraining with model {}'.format(model_mode)
    source_acc, target_acc = train_and_evaluate(graph, model)
    print '\nSource (MNIST) accuracy:', source_acc
    print '\nTarget (MNIST-M) accuracy:', target_acc








