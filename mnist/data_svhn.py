import sys
import os
from six.moves import urllib
from scipy.io import loadmat
import numpy as np

def maybe_download(data_dir):
    new_data_dir = os.path.join(data_dir, 'svhn')
    if not os.path.exists(new_data_dir):
        os.makedirs(new_data_dir)
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %.1f%%' % (float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve('http://ufldl.stanford.edu/housenumbers/train_32x32.mat', new_data_dir+'/train_32x32.mat', _progress)
        filepath, _ = urllib.request.urlretrieve('http://ufldl.stanford.edu/housenumbers/test_32x32.mat', new_data_dir+'/test_32x32.mat', _progress)

def load_data(dataset):
    data_dir = 'data/' + dataset
    maybe_download(data_dir)

    train_data = loadmat(os.path.join(data_dir, 'svhn') + '/train_32x32.mat')
    trX = np.asarray(train_data['X'])
    trX = trX.transpose((3,0,1,2))
    print(trX.shape)
    trY = np.asarray(train_data['y'].flatten())
    trY[trY==10] = 0
    trY_vec = np.zeros((len(trY), 10), dtype=np.float)
    for i, label in enumerate(trY):
        trY_vec[i,trY[i]] = 1.0


    test_data = loadmat(os.path.join(data_dir, 'svhn') + '/test_32x32.mat')
    teX = np.asarray(test_data['X'])
    teX = teX.transpose((3,0,1,2))
    teY = np.asarray(test_data['y'].flatten())
    teY[teY==10] = 0
    teY_vec = np.zeros((len(teY), 10), dtype=np.float)
    for i, label in enumerate(teY):
        teY_vec[i,teY[i]] = 1.0

    return trX/255., trY_vec, teX/255., teY_vec



