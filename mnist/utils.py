"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from six.moves import xrange
import subprocess

import tensorflow as tf
import tensorflow.contrib.slim as slim

import os

pp = pprint.PrettyPrinter()

def load_data(dataset):
    if dataset in ['mnist', 'fashion']:
        import data_mnist as file_load
    elif dataset == 'svhn':
        import data_svhn as file_load
    elif dataset == 'cifar10':
        import data_cifar10 as file_load
    else:
        raise NotImplementedError

    return file_load.load_data(dataset)

def evaluate(classifier, model, config):

    pred_prob = classifier.predict(model.sample())

    for i in range(50000 // model.batch_size):
        pred_prob = np.concatenate((pred_prob, classifier.predict(model.sample())))

    pred_prob = np.maximum(pred_prob, 1e-20*np.ones_like(pred_prob))
    # analyze label distribution

    y_vec = 1e-20 * np.ones((len(pred_prob), 10), dtype=np.float) # pred label distr
    gnd_vec = 0.1 * np.ones((1,10), dtype=np.float) # gnd label distr, uniform over 10 digits

    for i, label in enumerate(pred_prob):
        y_vec[i,np.argmax(pred_prob[i])] = 1.0
    y_vec = np.sum(y_vec, axis=0, keepdims=True)
    y_vec = y_vec / np.sum(y_vec)

    #label_entropy = np.sum(-y_vec * np.log(y_vec)).tolist()
    label_tv = np.true_divide(np.sum(np.abs(y_vec - gnd_vec)), 2).tolist()
    #label_l2 = np.sum((y_vec - gnd_vec)**2).tolist()
    y_vec = y_vec.tolist()

    print('TV: {}'.format(label_tv))

    return label_tv, y_vec

def compute_pairwise_distances(x, y):
    if not len(x.get_shape()) == len(y.get_shape()) == 2:
        raise ValueError('Both inputs should be matrices.')
    if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
        raise ValueError('The number of features should be the same.')
    norm = lambda x: tf.reduce_sum(tf.square(x), 1)

    return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))

def gaussian_kernel_matrix(x, y, sigmas):
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))
    dist = compute_pairwise_distances(x, y)
    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))
    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')

def inverse_transform(images):
    return (images+1.)/2.

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def pullaway_loss(embeddings):
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    similarity = tf.matmul(normalized_embeddings, normalized_embeddings, transpose_b=True)
    batch_size = tf.cast(tf.shape(embeddings)[0], tf.float32)
    pt_loss = (tf.reduce_sum(similarity) - batch_size) / (batch_size * (batch_size - 1))
    return pt_loss



