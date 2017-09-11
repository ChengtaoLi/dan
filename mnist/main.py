import os
import pickle

import scipy.misc
import numpy as np
import tensorflow as tf

import model_mlp
import model_conv

import mnist_classifier
import utils
from utils import pp

flags = tf.app.flags

# model settings

flags.DEFINE_string(
    'model_mode', 'dan_s', 
    'type of model from {gan, ebgan, wgan, wgan_gp, dan_s, dan_2s} [dan_s]')
flags.DEFINE_string(
    'model_layer', 'mlp',
    'type of model layer from {mlp, conv}')

# training settings

flags.DEFINE_string(
    'dataset', 'mnist', 'dataset from {mnist, fashion_mnist}')
flags.DEFINE_integer(
    'epoch', 50, 'number of epochs to train [50]')
flags.DEFINE_float(
    'lr', 0.0005, 'learning rate of for adam [0.0005]')
flags.DEFINE_float(
    'beta1', 0.5, 'momentum term of adam [0.5]')
flags.DEFINE_string(
    'ckpt_dir', 'ckpt', 'directory name to save the checkpoints [ckpt]')
flags.DEFINE_string(
    'smpl_dir', 'smpl', 'directory name to save the image samples [smpl]')
flags.DEFINE_boolean(
    'flag_train', True, 'True for training [True]')
flags.DEFINE_boolean(
    'flag_infer', True, 'True for generating samples [True]')
flags.DEFINE_boolean(
    'flag_classify', False, 'True for classifying [False]')
flags.DEFINE_string(
    'savepath', 'eval.txt', 'save path for evaluation [eval.txt]')

config = flags.FLAGS

def main(_):
    pp.pprint(config.__flags)

    with tf.Session() as sess:
        if config.model_layer == 'mlp':
            model = model_mlp.AdversarialNet(sess, config)
        elif config.model_layer == 'conv':
            model = model_conv.AdversarialNet(sess, config)
        else:
            raise NotImplementedError

        # train a model
        if config.flag_train:
            model.train(config)
        
        if config.flag_infer:
            if not model.load(config.ckpt_dir):
                raise Exception('[!] Train a model first, then do inference')

            model.infer()
            
        # classify generated samples and check label distribution
        if config.flag_classify:
            if not model.load(config.ckpt_dir):
                raise Exception('[!] Train a model first, then run test mode')

            model.classifier.load('mnist_cnn')
            pred_prob = model.classifier.predict(
                np.reshape(model.sample(), (model.batch_size, 784))
            )

            for i in range(100000 // model.batch_size):
                pred_prob = np.concatenate((
                    pred_prob,
                    model.classifier.predict(
                        np.reshape(model.sample_classify(), (model.batch_size, 784)))
                ))

            pred_prob = np.maximum(pred_prob, 1e-20*np.ones_like(pred_prob))
            # analyze label distribution

            y_vec = 1e-20 * np.ones((len(pred_prob), 10), dtype=np.float) # pred label distr
            gnd_vec = 0.1 * np.ones((1,10), dtype=np.float) # gnd label distr, uniform over 10 digits

            for i, label in enumerate(pred_prob):
                y_vec[i,np.argmax(pred_prob[i])] = 1.0
            y_vec = np.sum(y_vec, axis=0, keepdims=True)
            y_vec = y_vec / np.sum(y_vec)

            label_entropy = np.sum(-y_vec * np.log(y_vec)).tolist()
            label_tv = np.true_divide(np.sum(np.abs(y_vec - gnd_vec)), 2).tolist()
            label_l2 = np.sum((y_vec - gnd_vec)**2).tolist()

            pickle.dump({
                'entropy': label_entropy,
                'tv': label_tv,
                'l2': label_l2
            }, open(config.savepath, 'wb'))

            print('Entropy: {}'.format(label_entropy))
            print('TV: {}'.format(label_tv))
            print('L2: {}'.format(label_l2))

if __name__ == '__main__':
    utils.download_mnist()
    tf.app.run()
