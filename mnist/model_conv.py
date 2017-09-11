from __future__ import division

import os
import time
import math

import tensorflow as tf
import numpy as np
from six.moves import xrange

import ops
import utils
import mnist_classifier

class AdversarialNet(object):
    def __init__(self, sess, config):

        # model settings
        self.sess = sess
        self.classifier = mnist_classifier.mnist_cnn(sess)
        self.model_mode = config.model_mode
        self.dataset = config.dataset

        self.z_dim = config.z_dim
        self.batch_size = config.batch_size
        self.sample_num = config.batch_size

        self.ckpt_dir = config.ckpt_dir

        if self.dataset == 'mnist' or self.dataset == 'fashion_mnist':
            self.input_height = 28
            self.input_width = 28
            self.output_height = 28
            self.output_width = 28

        self.build_model()

    def build_model(self):

        image_dims = [self.input_height, self.input_width]

        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='real_images')
        self.z = tf.placeholder(
            tf.float32, [self.batch_size, self.z_dim], name='z')

        # generator
        self.G = self.generator(self.z)

        # discriminator
        self.D_real, self.D_logits_real = self.discriminator(self.inputs)
        self.D_fake, self.D_logits_fake = self.discriminator(self.G, flag_reuse=True)

        # sampler
        self.sampler = self.generator(self.z, flag_reuse=True)

        # losses
        self.g_loss = 0
        self.d_loss = 0
        self.s_loss = None

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits = self.D_logits_real,
                labels = tf.ones_like(self.D_real)
            ))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits = self.D_logits_fake,
                labels = tf.zeros_like(self.D_fake)
            ))
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits = self.D_logits_fake,
                labels = tf.ones_like(self.D_fake)
            ))

        # distributional adversary

        if self.model_mode == 'dan_s':
            self.D_set_real, self.D_set_logits_real = \
                    self.discriminator_dan_s(self.inputs)
            self.D_set_fake, self.D_set_logits_fake = \
                    self.discriminator_dan_s(self.G, flag_reuse=True)

            self.s_loss_real = tf.reduce_mean(
              tf.nn.sigmoid_cross_entropy_with_logits(
                  logits = self.D_set_logits_real,
                  labels = tf.ones_like(self.D_set_real)
              ))
            self.s_loss_fake = tf.reduce_mean(
              tf.nn.sigmoid_cross_entropy_with_logits(
                  logits = self.D_set_logits_fake,
                  labels = tf.zeros_like(self.D_set_fake)
              ))
            self.s_loss = self.s_loss_real + self.s_loss_fake

            self.g_loss += 0.2 * tf.reduce_mean(
              tf.nn.sigmoid_cross_entropy_with_logits(
                  logits = self.D_set_logits_fake,
                  labels = tf.ones_like(self.D_set_fake)
              ))

        elif self.model_mode == 'dan_2s' :
            self.D_11, self.D_00, self.D_10, self.D_01 = self.discriminator_dan_2s(self.inputs, self.G)

            self.s_loss_11 = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits = self.D_11,
                    labels = tf.ones_like(self.D_11)
                ))
            self.s_loss_00 = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits = self.D_00,
                    labels = tf.ones_like(self.D_00)
                ))
            self.s_loss_10 = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits = self.D_10,
                    labels = tf.zeros_like(self.D_10)
                ))
            self.s_loss_01 = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits = self.D_01,
                    labels = tf.zeros_like(self.D_01)
                ))

            self.s_loss = self.s_loss_11 + self.s_loss_00 + self.s_loss_10 + self.s_loss_01
            self.g_loss += 0.1*tf.reduce_mean(
              tf.nn.sigmoid_cross_entropy_with_logits(
                  logits = self.D_10,
                  labels = tf.ones_like(self.D_10)
              )) + 0.1 * tf.reduce_mean(
              tf.nn.sigmoid_cross_entropy_with_logits(
                  logits = self.D_01,
                  labels = tf.ones_like(self.D_01)
              ))

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        self.s_vars = [var for var in t_vars if 's_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        """Train model"""
        data, labels, test_data, test_labels = utils.load_mnist()
        np.random.shuffle(data)

        g_optim = tf.train.AdamOptimizer(
            config.learning_rate, beta1=config.beta1
        ).minimize(self.g_loss, var_list=self.g_vars)

        d_optim = tf.train.AdamOptimizer(
            config.learning_rate, beta1=config.beta1
        ).minimize(self.d_loss, var_list=self.d_vars)

        if self.s_loss is not None:
            s_optim = tf.train.AdamOptimizer(
                config.learning_rate, beta1=config.beta1
            ).minimize(self.s_loss, var_list=self.s_vars)


        tf.global_variables_initializer().run()

        sample_z = np.random.uniform(-1, 1, size=[self.sample_num, self.z_dim])

        counter = 0
        start_time = time.time()
        could_load, ckpt_counter = self.load(self.ckpt_dir)
        if could_load:
            counter = ckpt_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(config.epoch):
            batch_idxs = len(data) // self.batch_size
            for idx in xrange(0, batch_idxs):
                batch_images = data[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_labels = labels[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_z = np.random.uniform(-1, 1, size=[self.batch_size, self.z_dim])

                curr_d_loss, curr_g_loss, curr_s_loss = 0.0, 0.0, 0.0

                # Update D network
                _, curr_d_loss = self.sess.run(
                    [d_optim, self.d_loss],
                    feed_dict={
                        self.inputs: batch_images,
                        self.z: batch_z
                    }
                )

                # Update DAN every 5 iters
                if self.s_loss is not None:
                    if np.mod(counter+1, 5) == 0:
                        _, curr_s_loss = self.sess.run(
                            [s_optim, self.s_loss],
                            feed_dict={
                                self.inputs: batch_images,
                                self.z: batch_z
                            }
                        )
                    else:
                        curr_s_loss = self.sess.run(
                            self.s_loss, feed_dict={
                                self.inputs: batch_images,
                                self.z: batch_z
                            }
                        )

                # Update G network
                _, curr_g_loss = self.sess.run(
                    [g_optim, self.g_loss],
                    feed_dict={
                        self.inputs: batch_images,
                        self.z: batch_z
                    })

                counter += 1

                print("Epoch: [%2d] [%4d/%4d], time: %4.4f, d_loss: %.8f, g_loss: %.8f, s_loss: %.8f" \
                      % (epoch, idx, batch_idxs, time.time() - start_time, \
                         curr_d_loss, curr_g_loss, curr_s_loss))

            samples = self.sess.run(self.sampler, feed_dict={ self.z: sample_z})

            self.save_smpl(config.smpl_dir, epoch, samples)
            self.save(config.ckpt_dir, epoch)

    def infer(self):
        batch_z = np.random.uniform(-1, 1, size=[self.batch_size, self.z_dim])
        samples = self.sess.run(self.sampler, feed_dict={ self.z: batch_z })

        samples = np.reshape(samples, (-1,28,28,1))
        manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
        manifold_w = int(np.floor(np.sqrt(samples.shape[0])))

        smpl_dir = os.path.join('infer/', self.model_dir)
        if not os.path.exists(smpl_dir):
            os.makedirs(smpl_dir)

        utils.save_images(samples, [manifold_h, manifold_w],
                          './{}/infer.png'.format(smpl_dir))
        
    def sample(self):
        batch_z = np.random.uniform(-1, 1, size=[self.batch_size, self.z_dim])
        samples = self.sess.run(self.sampler, feed_dict={ self.z: batch_z })

        return samples


    def discriminator_dan_2s(self, image_real, image_fake, flag_reuse=False):
        with tf.variable_scope("discriminator_dan_2s", reuse=flag_reuse):
            image_real_0, image_real_1 = tf.split(image_real, num_or_size_splits=2, axis=0)
            image_fake_0, image_fake_1 = tf.split(image_fake, num_or_size_splits=2, axis=0)


            h_real_0_0 = ops.lrelu(ops.conv2d(image_real_0, 64, 4, 4, 2, 2, name='s_conv0'))
            h_real_1_0 = ops.lrelu(ops.conv2d(image_real_1, 64, 4, 4, 2, 2, name='s_conv0', reuse=True))
            h_fake_0_0 = ops.lrelu(ops.conv2d(image_fake_0, 64, 4, 4, 2, 2, name='s_conv0', reuse=True))
            h_fake_1_0 = ops.lrelu(ops.conv2d(image_fake_1, 64, 4, 4, 2, 2, name='s_conv0', reuse=True))

            h_real_0_1 = ops.lrelu(ops.conv2d(h_real_0_0, 128, 4, 4, 2, 2, name='s_conv1'))
            h_real_1_1 = ops.lrelu(ops.conv2d(h_real_1_0, 128, 4, 4, 2, 2, name='s_conv1', reuse=True))
            h_fake_0_1 = ops.lrelu(ops.conv2d(h_fake_0_0, 128, 4, 4, 2, 2, name='s_conv1', reuse=True))
            h_fake_1_1 = ops.lrelu(ops.conv2d(h_fake_1_0, 128, 4, 4, 2, 2, name='s_conv1', reuse=True))

            h_real_0_avg = tf.reduce_mean(tf.reshape(h_real_0_1, [int(self.batch_size/2), -1]), axis=0, keep_dims=True)
            h_real_1_avg = tf.reduce_mean(tf.reshape(h_real_1_1, [int(self.batch_size/2), -1]), axis=0, keep_dims=True)
            h_fake_0_avg = tf.reduce_mean(tf.reshape(h_fake_0_1, [int(self.batch_size/2), -1]), axis=0, keep_dims=True)
            h_fake_1_avg = tf.reduce_mean(tf.reshape(h_fake_1_1, [int(self.batch_size/2), -1]), axis=0, keep_dims=True)

            h_11_2 = ops.lrelu(ops.linear(tf.abs(h_real_0_avg - h_real_1_avg), 1024, 's_fc2'))
            h_00_2 = ops.lrelu(ops.linear(tf.abs(h_fake_0_avg - h_fake_1_avg), 1024, 's_fc2', reuse=True))
            h_10_2 = ops.lrelu(ops.linear(tf.abs(h_real_0_avg - h_fake_1_avg), 1024, 's_fc2', reuse=True))
            h_01_2 = ops.lrelu(ops.linear(tf.abs(h_fake_0_avg - h_real_1_avg), 1024, 's_fc2', reuse=True))

            h_11_fin = ops.linear(h_11_2, 1, 's_fin')
            h_00_fin = ops.linear(h_00_2, 1, 's_fin', reuse=True)
            h_10_fin = ops.linear(h_10_2, 1, 's_fin', reuse=True)
            h_01_fin = ops.linear(h_01_2, 1, 's_fin', reuse=True)

            return h_11_fin, h_00_fin, h_10_fin, h_01_fin

    def discriminator_dan_s(self, image, flag_train=True, flag_reuse=False):
        with tf.variable_scope("discriminator_dan_s", reuse=flag_reuse):
            h_0 = ops.lrelu(ops.conv2d(image, 64, 4, 4, 2, 2, name='s_conv0'))
            h_1 = ops.lrelu(ops.conv2d(h_0, 128, 4, 4, 2, 2, name='s_conv1'))
            h_1_avg = tf.reduce_mean(tf.reshape(h_1, [self.batch_size, -1]), axis=0, keep_dims=True)
            h_2 = ops.lrelu(ops.linear(h_1_avg, 1024, scope='s_fc2'))
            h_fin = ops.linear(h_2, 1, scope='s_fin')

            return tf.nn.sigmoid(h_fin), h_fin

    def discriminator(self, image, flag_train=True, flag_reuse=False):
        with tf.variable_scope("discriminator", reuse=flag_reuse):
            h_0 = ops.lrelu(ops.conv2d(image, 64, 4, 4, 2, 2, name='d_conv0'))
            h_1 = ops.lrelu(ops.bn(ops.conv2d(h_0, 128, 4, 4, 2, 2, name='d_conv1'), is_training=flag_train, scope='d_bn1'))
            h_1_flat = tf.reshape(h_1, [self.batch_size, -1])
            h_2 = ops.lrelu(ops.bn(ops.linear(h_1_flat, 1024, scope='d_fc2'), is_training=flag_train, scope='d_bn2'))
            h_fin = ops.linear(h_2, 1, scope='d_fin')

            return tf.nn.sigmoid(h_fin), h_fin

    def generator(self, z, flag_train=True, flag_reuse=False):
        with tf.variable_scope("generator", reuse=flag_reuse):
            h_0 = tf.nn.relu(ops.bn(ops.linear(z, 1024, scope='g_fc0'), is_training=is_training, scope='g_bn0'))
            h_1 = tf.nn.relu(ops.bn(ops.linear(h_0, 128 * 7 * 7, scope='g_fc1'), is_training=is_training, scope='g_bn1'))
            h_1_flat = tf.reshape(h_1, [self.batch_size, 7, 7, 128])
            h_2 = tf.nn.relu(ops.bn(
                ops.deconv2d(h_1_flat, [self.batch_size, 14, 14, 64], 4, 4, 2, 2, name='g_dc2'), 
                is_training=is_training, scope='g_bn2'
            ))
            h_fin = deconv2d(h_2, [self.batch_size, 28, 28, 1], 4, 4, 2, 2, name='g_dc3')

            return tf.nn.sigmoid(h_fin)

    @property
    def model_dir(self):
        return self.model_mode + '-conv-' + self.dataset

    def save_smpl(self, smpl_dir, epoch, samples):
        samples = np.reshape(samples, (-1,28,28,1))
        manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
        manifold_w = int(np.floor(np.sqrt(samples.shape[0])))

        smpl_dir = os.path.join(smpl_dir, self.model_dir)
        if not os.path.exists(smpl_dir):
            os.makedirs(smpl_dir)

        utils.save_images(samples, [manifold_h, manifold_w],
                          './{}/train_{:04d}.png'.format(smpl_dir, epoch))

    def save(self, ckpt_dir, step):
        model_name = "AdversarialNet.model"
        ckpt_dir = os.path.join(ckpt_dir, self.model_dir)

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        self.saver.save(
            self.sess,
            os.path.join(ckpt_dir, model_name),
            global_step=step)

    def load(self, ckpt_dir):
        import re
        print(" [*] Reading checkpoints...")
        ckpt_dir = os.path.join(ckpt_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(ckpt_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
