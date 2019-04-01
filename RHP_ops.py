import tensorflow as tf
from tensorpack.models import Conv2D

from region_norm_ops import get_rn


def conv_with_rn(gradient):
    out = Conv2D('conv', gradient, gradient.get_shape()[3], 1, strides=1, activation=get_rn(),
                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0))
    gradient = gradient + out
    return gradient
