import os
import pickle

import scipy.misc
import numpy as np
import tensorflow as tf

import model_mnist_mlp
import mnist_classifier
from utils import pp

flags = tf.app.flags

# model settings

flags.DEFINE_string(
    "model_mode", "gan", "type of model [gan, dan_s, dan_2s]")

# training settings

flags.DEFINE_integer(
    "epoch", 100, "Epoch to train [100]")
flags.DEFINE_float(
    "learning_rate", 0.0005, "Learning rate of for adam [0.0005]")
flags.DEFINE_float(
    "beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_string(
    "ckpt_dir", "ckpt_mnist", "Directory name to save the checkpoints [ckpt_mnist]")
flags.DEFINE_string(
    "smpl_dir", "smpl_mnist", "Directory name to save the image samples [smpl_mnist]")
flags.DEFINE_boolean(
    "is_train", True, "True for training, False for evaluating [True]")
flags.DEFINE_string(
    "savepath", "eval.txt", "save path for evaluation [eval.txt]")

config = flags.FLAGS

def main(_):
    pp.pprint(config.__flags)

    with tf.Session() as sess:
        model = model_mnist_mlp.AdversarialNet(sess, config)

        if config.is_train:
            model.train(config)
        else:
            if not model.load(config.ckpt_dir):
                raise Exception("[!] Train a model first, then run test mode")

            model.classifier.load("mnist_cnn")
            pred_prob = model.classifier.predict(
                np.reshape(model.sample_classify(), (model.batch_size, 784))
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

            print("Entropy: {}".format(label_entropy))
            print("TV: {}".format(label_tv))
            print("L2: {}".format(label_l2))

if __name__ == '__main__':
    tf.app.run()
