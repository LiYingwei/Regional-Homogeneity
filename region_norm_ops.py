import tensorflow as tf
from tensorpack.models import BatchNorm, LayerNorm, InstanceNorm, layer_register, VariableHolder
from tensorpack.utils.argtools import get_data_format
import numpy as np

from config import config as FLAGS


def get_rn(zero_init=FLAGS.zero_init):
    """
    Zero init gamma is good for resnet. See https://arxiv.org/abs/1706.02677.
    """
    if zero_init:
        return lambda x, name=None: RegionNorm('rn', x, h_group_num=FLAGS.h_group_num,
                                               w_group_num=FLAGS.w_group_num,
                                               gamma_initializer=tf.zeros_initializer())
    else:
        return lambda x, name=None: RegionNorm('rn', x, h_group_num=FLAGS.h_group_num,
                                               w_group_num=FLAGS.w_group_num)


def get_pad_num(total_length, group_num):
    remain = group_num - total_length % group_num if total_length % group_num != 0 else 0
    return remain + total_length


@layer_register(log_shape=True)
def RegionNorm(x, h_group_num, w_group_num, gamma_initializer=tf.constant_initializer(1.)):
    # 1. pad so that h % h_group_num == 0, w % w_group_num == 0
    orig_shape = x.get_shape().as_list()
    h, w = orig_shape[1], orig_shape[2]
    new_h = get_pad_num(h, h_group_num)
    new_w = get_pad_num(w, w_group_num)
    x_resized = tf.image.resize_images(x, [new_h, new_w], align_corners=False)

    # 2. split and stack all grid
    assert new_h % h_group_num == 0
    sub_h = new_h // h_group_num
    assert new_w % w_group_num == 0
    sub_w = new_w // w_group_num

    sub_grids = []
    for i in range(0, new_h, sub_h):
        for j in range(0, new_w, sub_w):
            x_sub_grid = x_resized[:, i:i + sub_h, j:j + sub_w, :, None]
            sub_grids.append(x_sub_grid)

    sub_grids = tf.concat(sub_grids, axis=4)
    sub_grids_shape = sub_grids.get_shape().as_list()
    feed2bn = tf.reshape(sub_grids,
                         [-1, sub_grids_shape[1], sub_grids_shape[2] * sub_grids_shape[3],
                          sub_grids_shape[4]])
    # 3. normalization
    bn_output = BatchNorm('bn', feed2bn, axis=3, gamma_initializer=gamma_initializer,
                          internal_update=True, sync_statistics='nccl')
    # 4. go back to original shape
    new_sub_grids = tf.reshape(bn_output,
                               [-1, sub_grids_shape[1], sub_grids_shape[2], sub_grids_shape[3],
                                sub_grids_shape[4]])
    counter = 0
    new_rows = []
    for i in range(0, new_h, sub_h):
        new_row = []
        for j in range(0, new_w, sub_w):
            new_row.append(new_sub_grids[:, :, :, :, counter])
            counter += 1
        new_row = tf.concat(new_row, axis=2)
        new_rows.append(new_row)
    new_x_resized = tf.concat(new_rows, axis=1)
    # 5. resize back
    new_x = tf.image.resize_images(new_x_resized, [h, w], align_corners=False)
    return new_x
