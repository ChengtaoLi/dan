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

def maybe_download(data_dir):
    if os.path.exists(data_dir):
        print('Found ' + data_dir + ' - skip')
        return
    else:
        os.mkdir(data_dir)

    if data_dir.endswith('mnist'):
        url_base = 'http://yann.lecun.com/exdb/mnist/'

    elif data_dir.endswith('fashion'):
        url_base = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'

    else:
        raise NotImplementedError

    file_names = ['train-images-idx3-ubyte.gz',
                  'train-labels-idx1-ubyte.gz',
                  't10k-images-idx3-ubyte.gz',
                  't10k-labels-idx1-ubyte.gz']

    for file_name in file_names:
        url = (url_base+file_name).format(**locals())
        print(url)
        out_path = os.path.join(data_dir,file_name)
        cmd = ['curl', url, '-o', out_path]
        print('Downloading ', file_name)
        subprocess.call(cmd)
        cmd = ['gzip', '-d', out_path]
        print('Decompressing ', file_name)
        subprocess.call(cmd)

def load_data(dataset):
    data_dir = 'data/' + dataset
    maybe_download(data_dir)

    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.int)

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

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


