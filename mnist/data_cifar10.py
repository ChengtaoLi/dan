import cPickle
import os
import sys
import tarfile
from six.moves import urllib
import numpy as np

def maybe_download_and_extract(data_dir, url='http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'):
    if not os.path.exists(os.path.join(data_dir, 'cifar-10-batches-py')):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        filename = url.split('/')[-1]
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                    float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
            tarfile.open(filepath, 'r:gz').extractall(data_dir)

def unpickle(file):
    fo = open(file, 'rb')
    d = cPickle.load(fo)
    fo.close()
    return {'x': np.cast[np.float32]((d['data'].reshape((10000,3,32,32)))/255.), 'y': np.array(d['labels']).astype(np.int32)}

def load_data(dataset):
    data_dir = 'data/' + dataset

    maybe_download_and_extract(data_dir)

    train_data = [unpickle(os.path.join(data_dir,'cifar-10-batches-py/data_batch_' + str(i))) for i in range(1,6)]
    trX = np.concatenate([d['x'] for d in train_data],axis=0).transpose((0,2,3,1))
    trY = np.concatenate([d['y'] for d in train_data],axis=0)

    trY_vec = np.zeros((len(trY), 10), dtype=np.float)
    for i, label in enumerate(trY):
        trY_vec[i,trY[i]] = 1.0

    test_data = unpickle(os.path.join(data_dir,'cifar-10-batches-py/test_batch'))
    teX = test_data['x'].transpose((0,2,3,1))
    teY = test_data['y']

    teY_vec = np.zeros((len(teY), 10), dtype=np.float)
    for i, label in enumerate(teY):
        teY_vec[i,teY[i]] = 1.0

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(trX)
    np.random.seed(seed)
    np.random.shuffle(trY)
    np.random.seed(seed)
    np.random.shuffle(teX)
    np.random.seed(seed)
    np.random.shuffle(teY)

    return trX, trY_vec, teX, teY_vec




