'''
Slightly modified base on:
https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import tflearn

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, global_avg_pool
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import batch_normalization, local_response_normalization

import utils
import os

import data_mnist
import data_svhn
import data_cifar10

class classifier_net(object):
    def __init__(self, dataset):
        self.dataset = dataset

        if dataset in ['mnist', 'fashion']:
            network = input_data(shape=[None, 28, 28, 1], name='input')
            network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
            network = max_pool_2d(network, 2)
            network = local_response_normalization(network)
            network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
            network = max_pool_2d(network, 2)
            network = local_response_normalization(network)
            network = fully_connected(network, 128, activation='tanh')
            network = dropout(network, 0.8)
            network = fully_connected(network, 256, activation='tanh')
            network = dropout(network, 0.8)
            network = fully_connected(network, 10, activation='softmax')
            network = regression(network, optimizer='adam', learning_rate=0.001,
                                         loss='categorical_crossentropy', name='target')

        elif dataset == 'svhn':
            network = input_data(shape=[None, 32, 32, 3], name='input')
            network = conv_2d(network, 32, 3, activation='relu')
            network = max_pool_2d(network, 2)
            network = conv_2d(network, 64, 3, activation='relu')
            network = conv_2d(network, 64, 3, activation='relu')
            network = max_pool_2d(network, 2)
            network = fully_connected(network, 512, activation='relu')
            network = dropout(network, 0.5)
            network = fully_connected(network, 10, activation='softmax')
            network = regression(network, optimizer='adam', learning_rate=0.001,
                    loss='categorical_crossentropy', name='target')

        elif dataset == 'cifar10':
            network = input_data(shape=[None, 32, 32, 3], name='input')

            '''
            n=5
            network = tflearn.conv_2d(network, 16, 3, regularizer='L2', weight_decay=0.0001)
            network = tflearn.residual_block(network, n, 16)
            network = tflearn.residual_block(network, 1, 32, downsample=True)
            network = tflearn.residual_block(network, n-1, 32)
            network = tflearn.residual_block(network, 1, 64, downsample=True)
            network = tflearn.residual_block(network, n-1, 64)
            network = tflearn.batch_normalization(network)
            network = tflearn.activation(network, 'relu')
            network = tflearn.global_avg_pool(network)
            # Regression
            network = tflearn.fully_connected(network, 10, activation='softmax')
            mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
            network = tflearn.regression(network, optimizer=mom,
                                             loss='categorical_crossentropy')
            # Training
            model = tflearn.DNN(network, tensorboard_verbose=0)
            model.fit({'input': trX}, {'target': trY}, n_epoch=200,
                    validation_set=({'input': teX}, {'target': teY}),
                    snapshot_step=100, show_metric=True, run_id='net_cifar10')
            '''

            network = conv_2d(network, 32, 3, activation='relu')
            network = max_pool_2d(network, 2)
            network = conv_2d(network, 64, 3, activation='relu')
            network = conv_2d(network, 64, 3, activation='relu')
            network = max_pool_2d(network, 2)
            network = fully_connected(network, 512, activation='relu')
            network = dropout(network, 0.5)
            network = fully_connected(network, 10, activation='softmax')
            network = regression(network, optimizer='adam', learning_rate=0.001,
                    loss='categorical_crossentropy', name='target')

        # Training
        self.model = tflearn.DNN(network, tensorboard_verbose=0)

    def train(self):
        # tf Graph input
        trX, trY, teX, teY = utils.load_data(self.dataset)

        if self.dataset in ['mnist', 'fashion', 'svhn']:
            self.model.fit({'input': trX}, {'target': trY}, n_epoch=20,
                    validation_set=({'input': teX}, {'target': teY}),
                    snapshot_step=100, show_metric=True, run_id='net_'+self.dataset)
            self.model.save('trained/net_'+self.dataset+'.tfl')

    def predict(self, x):
        pred_y = self.model.predict(x)

        return pred_y

    def test(self, x, y):
        pred_y = self.predict(x)
        correct_pred = tf.equal(tf.argmax(model.predict(teX[:5000]), 1), tf.argmax(teY[:5000], 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        return accuracy

    def load(self):
        self.model.load('trained/net_'+self.dataset+'.tfl')

if __name__ == '__main__':
    flags = tf.app.flags

    flags.DEFINE_string(
        'dataset', 'mnist', 'dataset from {mnist, fashion, svhn, cifar10}')

    model = classifier_net(flags.FLAGS.dataset)
    model.train()

