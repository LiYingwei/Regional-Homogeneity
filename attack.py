from config import config as FLAGS
import tensorflow as tf
from tensorpack import (BatchData)
from tqdm import tqdm
import numpy as np

from RHP_ops import conv_with_rn
from data import PNGDataFlow, save_images
from networks import network

from tensorpack.tfutils.tower import TowerContext


class Attacker:
    def __init__(self, sess):
        self.sess = sess
        self.step_size = FLAGS.step_size / 255.0
        self.max_epsilon = FLAGS.max_epsilon / 255.0
        # Prepare graph
        batch_shape = [FLAGS.batch_size, 299, 299, 3]
        self.x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_max = tf.clip_by_value(self.x_input + self.max_epsilon, 0., 1.0)
        x_min = tf.clip_by_value(self.x_input - self.max_epsilon, 0., 1.0)

        self.y_input = tf.placeholder(tf.int64, shape=batch_shape[0])
        i = tf.constant(0)
        self.x_adv, _, _, _, _ = tf.while_loop(self.stop, self.graph,
                                               [self.x_input, self.y_input, i, x_max, x_min])
        self.restore()

    def graph(self, x, y, i, x_max, x_min):
        with TowerContext("model_tower", is_training=False):
            logits, _, endpoints = network.model(x, FLAGS.attack_networks[0])

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
        noise = tf.gradients(loss, x)[0] if not FLAGS.universal else tf.zeros_like(x)
        with TowerContext('RHP_tower', is_training=False):
            with tf.variable_scope('RHP'):
                noise = conv_with_rn(noise)
        noise = noise / (tf.reduce_mean(tf.abs(noise), [1, 2, 3], keepdims=True) + 1e-12)
        x = x + self.step_size * tf.sign(noise)
        x = tf.clip_by_value(x, x_min, x_max)
        i = tf.add(i, 1)
        return x, y, i, x_max, x_min

    @staticmethod
    def stop(x, y, i, x_max, x_min):
        return tf.less(i, FLAGS.num_steps)

    def perturb(self, images, labels):
        batch_size = images.shape[0]
        if batch_size < FLAGS.batch_size:
            pad_num = FLAGS.batch_size - batch_size
            pad_img = np.zeros([pad_num, 299, 299, 3])
            images = np.concatenate([images, pad_img])
            pad_label = np.zeros([pad_num])
            labels = np.concatenate([labels, pad_label])
        adv_images = sess.run(self.x_adv, feed_dict={self.x_input: images, self.y_input: labels})
        return adv_images[:batch_size]

    def restore(self):
        network.restore(self.sess, FLAGS.attack_networks[0])
        RHP_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='RHP')
        RHP_variables_saver = tf.train.Saver(RHP_variables)
        ckpt_filename = tf.train.latest_checkpoint(FLAGS.RHP_savepath)
        RHP_variables_saver.restore(sess, ckpt_filename)


if __name__ == '__main__':
    sess = tf.Session()
    model = Attacker(sess)
    df = PNGDataFlow(FLAGS.img_dir, FLAGS.test_list_filename, FLAGS.ground_truth_file,
                     result_dir=None, img_num=FLAGS.img_num)
    df = BatchData(df, FLAGS.batch_size, remainder=True)
    df.reset_state()

    total_batch = int((df.ds.img_num - 1) / FLAGS.batch_size) + 1
    for batch_index, (x_batch, y_batch, name_batch) in tqdm(enumerate(df), total=total_batch):
        advs = model.perturb(x_batch, y_batch)
        save_images(advs, name_batch, FLAGS.result_dir)
