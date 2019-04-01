import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorpack import (BatchData)
from tensorpack.tfutils.tower import TowerContext
from tqdm import tqdm

from config import config as FLAGS
from data import PNGDataFlow
from networks import network


class Evaluator:
    def __init__(self, sess):
        self.sess = sess
        # Prepare graph
        self.build_graph()
        self.restore()

    def build_graph(self):
        batch_shape = [None, 299, 299, 3]
        self.x_input = tf.placeholder(tf.float32, shape=batch_shape)
        self.y_input = tf.placeholder(tf.int64, shape=batch_shape[0])
        self.acc_list = []
        self.predictions = []
        with TowerContext("model_tower", is_training=False):
            for network_name in FLAGS.test_networks:
                acc, predictions = network.model(self.x_input, network_name, label=self.y_input)
                self.acc_list.append(acc)
                self.predictions.append(predictions)

    def eval(self, images, labels):
        accs, preds = self.sess.run([self.acc_list, self.predictions],
                                    feed_dict={self.x_input: images, self.y_input: labels})
        # # try below lines if OOM
        # accs = []
        # for acc_tensor in self.acc_list:
        #     accs.append(
        #         self.sess.run(acc_tensor, feed_dict={self.x_input: images, self.y_input: labels}))
        return np.array(accs), np.stack(preds)

    def restore(self):
        for network_name in FLAGS.test_networks:
            network.restore(self.sess, network_name)


class AvgMetric(object):
    def __init__(self, datashape):
        self.cnt = np.zeros(datashape)
        self.sum = 0.

    def update(self, sum, cnt=1):
        self.sum += sum
        self.cnt += cnt

    def get_status(self):
        return self.sum / self.cnt


class Collector(object):
    def __init__(self, list_num):
        self.lists = [[] for _ in range(list_num)]

    def update(self, lists):
        for i, l in enumerate(lists):
            self.lists[i].append(l)

    def get_status(self, axis_list, index=None):
        ret = []
        for i, axis in enumerate(axis_list):
            ret.append(np.concatenate(self.lists[i], axis=axis))
        if index is not None:
            for i, axis in enumerate(axis_list):
                if axis == 0:
                    ret[i] = ret[i][index]
                else:
                    assert axis == 1
                    ret[i] = ret[i][:, index]
        return ret


def build_in_eval():
    with tf.Session() as sess:
        model = Evaluator(sess)
        df = PNGDataFlow(FLAGS.result_dir, FLAGS.test_list_filename, FLAGS.ground_truth_file,
                         img_num=FLAGS.img_num)
        df = BatchData(df, FLAGS.batch_size, remainder=True)
        df.reset_state()

        avgMetric = AvgMetric(datashape=[len(FLAGS.test_networks)])
        total_batch = df.ds.img_num / FLAGS.batch_size
        for batch_index, (x_batch, y_batch, name_batch) in tqdm(enumerate(df), total=total_batch):
            acc, pred = model.eval(x_batch, y_batch)
            avgMetric.update(acc)

    return 1 - avgMetric.get_status()


if __name__ == '__main__':
    result = build_in_eval()
    # output
    line0 = " ".join(sys.argv)
    line1 = np.array2string(np.array(FLAGS.test_networks), separator=', ', max_line_width=np.inf)[
            1:-1]
    line2 = np.array2string(result, separator=', ', precision=4, max_line_width=np.inf)[1:-1]
    string_to_write = line0 + "\n" + line1 + "\n" + line2 + "\n"
    print(string_to_write)
