import os
import pickle

import scipy.misc
import numpy as np
import tensorflow as tf

import classifiers
import utils
from utils import pp


flags = tf.app.flags

flags.DEFINE_string(
    'network', 'mlp',
    'type of network from {mlp, conv} [mlp]')
flags.DEFINE_string(
    'model_mode', 'dan_s',
    'type of model from {gan, wganori, wgangp, mmd, dan_s, dan_2s} [dan_s]')
flags.DEFINE_string(
    'dataset', 'mnist', 'dataset from {mnist, fashion, svhn, cifar10}')

# training settings

flags.DEFINE_integer(
    'epoch', 100, 'number of epochs to train [100]')
flags.DEFINE_float(
    'lr', 0.0005, 'learning rate of for adam [0.0005]')
flags.DEFINE_float(
    'beta1', 0.5, 'momentum term of adam [0.5]')
flags.DEFINE_integer(
    'z_dim', 256, 'latent dimension [256]')
flags.DEFINE_integer(
    'batch_size', 256, 'batch size [256]')
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
flags.DEFINE_boolean(
    'flag_load', False, 'True for loading trained model before training [False]')
flags.DEFINE_string(
    'savepath', 'eval.txt', 'save path for evaluation [eval.txt]')

config = flags.FLAGS

if __name__ == '__main__':
    pp.pprint(config.__flags)

    if config.network == 'mlp':
        from model_mlp import AdversarialNet
    elif config.network == 'conv':
        from model_conv import AdversarialNet
    else:
        raise NotImplementedError

    classifier = classifiers.classifier_net(config.dataset)
    classifier.load()

    with tf.Session() as sess:

        model = AdversarialNet(sess, config)

        # train a model
        if config.flag_train:
            perform_tv, perform_time = model.train(config, classifier)

            print(perform_tv)
            print(perform_time)

            pickle.dump({
                'tv': perform_tv,
                'time': perform_time
            }, open(config.savepath, 'wb'))

        if config.flag_infer:
            if not model.load(config.ckpt_dir):
                raise Exception('[!] Train a model first, then do inference')

            model.infer()

        # classify generated samples and check label distribution
        if config.flag_classify:
            if not model.load(config.ckpt_dir):
                raise Exception('[!] Train a model first, then run test mode')

            label_tv, y_vec = utils.evaluate(model, config)

            pickle.dump({
                'label': y_vec,
                'tv': label_tv
            }, open(config.savepath, 'wb'))


