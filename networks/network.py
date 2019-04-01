import functools
import importlib

import tensorflow as tf
import tensorflow.contrib.slim as slim


def _get_model(reuse, arg_scope, func, network_name):
    @functools.wraps(func)
    def network_fn(images):
        with slim.arg_scope(arg_scope):
            return func(images, 1001, is_training=False, reuse=reuse, scope=network_name)

    return network_fn


def _preprocess(image):
    return image * 2. - 1.


def _preprocess_resize(image, size):
    return tf.image.resize_images(image, [size, size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


def center_crop(image, size):
    image_height = tf.shape(image)[1]
    image_width = tf.shape(image)[2]

    offset_height = (image_height - size) // 2
    offset_width = (image_width - size) // 2
    image = tf.slice(image, [0, offset_height, offset_width, 0], [-1, size, size, -1])
    return image


def _preprocess_tp(image, bgr=True, data_format='NCHW'):
    image *= 255.0
    image = tf.image.resize_nearest_neighbor(image, [256, 256])
    image = center_crop(image, 224)
    # image = tf.image.resize_nearest_neighbor(image, [224, 224])

    mean = [0.485, 0.456, 0.406]  # rgb
    std = [0.229, 0.224, 0.225]
    if bgr:
        image = tf.reverse(image, axis=[3])  # to bgr
        mean = mean[::-1]
        std = std[::-1]
    image_mean = tf.constant(mean, dtype=tf.float32) * 255
    image_std = tf.constant(std, dtype=tf.float32) * 255
    image = (image - image_mean) / image_std

    if data_format == 'NCHW':
        image = tf.transpose(image, [0, 3, 1, 2])

    return image


def _preprocess_dataformat(image, bgr=True, data_format='NCHW'):
    if bgr:
        image = tf.reverse(image, axis=[3])  # to bgr

    if data_format == 'NCHW':
        image = tf.transpose(image, [0, 3, 1, 2])

    return image


_network_build = {}


def model(image, scope_name, label=None):
    # arg_scope, func, checkpoint_path
    network_core = importlib.import_module('networks.core.' + scope_name)

    if scope_name not in _network_build:
        _network_build[scope_name] = False

    network_fn = _get_model(reuse=tf.AUTO_REUSE, arg_scope=network_core.arg_scope, func=network_core.func, network_name=scope_name)
    _network_build[scope_name] = True
    if scope_name in ['webvision_resnet_18']:
        preprocessed = _preprocess_tp(image)
    else:
        preprocessed = _preprocess(image)
        if scope_name in ['resnet_v2_50_alp']:
            preprocessed = _preprocess_resize(preprocessed, 64)
        if scope_name in ['R152', 'R152-Denoise', 'X101-DenoiseAll']:
            preprocessed = _preprocess_resize(preprocessed, 224)
        if scope_name in ['webvision_resnet_v2_18', 'webvision_resnet_v2_50', 'R152', 'R152-Denoise', 'X101-DenoiseAll']:
            preprocessed = _preprocess_dataformat(preprocessed, bgr=True, data_format='NCHW')
    logits, end_points = network_fn(preprocessed)
    logits = tf.reshape(logits, shape=[-1, 1001])
    predictions = tf.argmax(logits, 1)
    if label is not None:
        acc = tf.reduce_mean(tf.cast(tf.equal(predictions, label), tf.float32))
        return acc, predictions

    return logits, predictions, end_points


_network_initialized = {}


def restore(sess, scope_name):
    network_core = importlib.import_module('networks.core.' + scope_name)
    global _network_initialized

    if (scope_name not in _network_initialized) or (not _network_initialized[scope_name]):
        ckpt_path = network_core.checkpoint_path
        optimistic_restore(sess, ckpt_path)
        _network_initialized[scope_name] = True


def optimistic_restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                        if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            if saved_var_name == 'global_step':
                continue
            curr_var = tf.get_variable(saved_var_name)
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
            else:
                print(var_shape, saved_shapes[saved_var_name], saved_var_name)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)
