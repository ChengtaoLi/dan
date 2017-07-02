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

import tensorflow as tf
import tensorflow.contrib.slim as slim

import os

pp = pprint.PrettyPrinter()

def load_mnist():
    data_dir = "./data/mnist"

    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,784)).astype(np.float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.int)

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,784)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.int)

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(trX)
    np.random.seed(seed)
    np.random.shuffle(trY)
    np.random.seed(seed)
    np.random.shuffle(teX)
    np.random.seed(seed)
    np.random.shuffle(teY)

    trY_vec = np.zeros((len(trY), 10), dtype=np.float)
    for i, label in enumerate(trY):
        trY_vec[i,trY[i]] = 1.0
    teY_vec = np.zeros((len(teY), 10), dtype=np.float)
    for i, label in enumerate(teY):
        teY_vec[i,teY[i]] = 1.0

    return trX/255., trY_vec, teX/255., teY_vec

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


