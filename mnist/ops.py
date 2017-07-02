import math

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

from utils import *

if "concat_v2" in dir(tf):
  def concat(tensors, axis, *args, **kwargs):
    return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
  def concat(tensors, axis, *args, **kwargs):
    return tf.concat(tensors, axis, *args, **kwargs)

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False, reuse=False):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(scope or "Linear", reuse=reuse):
    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias

