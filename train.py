import multiprocessing
import os

import tensorflow as tf
from tensorpack import *
from tensorpack.tfutils.varreplace import freeze_variables
from tensorpack.utils.gpu import get_num_gpu

from config import config as FLAGS
from data import PNGDataFlow
from networks import network
from RHP_ops import conv_with_rn


class Model(ModelDesc):
    def __init__(self):
        super(Model, self).__init__()

    def inputs(self):
        batch_shape = [None, 299, 299, 3]
        x_input = tf.placeholder(tf.float32, shape=batch_shape, name='image')
        y_input = tf.placeholder(tf.int64, shape=[batch_shape[0]], name='label')
        return [x_input, y_input]

    def build_graph(self, x, y):
        with freeze_variables(stop_gradient=False, skip_collection=True):
            step_size = FLAGS.step_size / 255.0
            max_epsilon = FLAGS.max_epsilon / 255.0
            x_max = tf.clip_by_value(x + max_epsilon, 0., 1.0)
            x_min = tf.clip_by_value(x - max_epsilon, 0., 1.0)

            logits, _, _ = network.model(x, FLAGS.attack_networks[0])
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
            noise = tf.gradients(loss, x)[0]

        with tf.variable_scope('RHP'):
            noise = conv_with_rn(noise)

        with freeze_variables(stop_gradient=False, skip_collection=True):
            G = tf.get_default_graph()
            with G.gradient_override_map({"Sign": "Identity"}):
                x = x + step_size * tf.sign(noise)
            x = tf.clip_by_value(x, x_min, x_max)

            # evaluate after add perturbation
            logits, _, _ = network.model(x, FLAGS.attack_networks[0])
            loss_to_optimize = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y),
                name='train_loss')

        return -loss_to_optimize

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=FLAGS.lr * (FLAGS.batch / 256.0),
                             trainable=False)
        if FLAGS.optimizer == 'momentum':
            opt = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
        else:
            assert FLAGS.optimizer == 'adam'
            opt = tf.train.AdamOptimizer(lr)
        return opt


class MultipleRestore(SessionInit):
    def __init__(self):
        pass

    def _setup_graph(self):
        pass

    def _run_init(self, sess):
        network.restore(sess, FLAGS.attack_networks[0])
        RHP_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='RHP')
        init_variables_op = tf.variables_initializer(RHP_variables)
        sess.run(init_variables_op)


class ScopeModelSaver(ModelSaver):
    def __init__(self, max_to_keep=0,
                 keep_checkpoint_every_n_hours=0.5,
                 checkpoint_dir=None,
                 var_collections=[tf.GraphKeys.GLOBAL_VARIABLES], scope=None):
        super(ScopeModelSaver, self).__init__(max_to_keep, keep_checkpoint_every_n_hours,
                                              checkpoint_dir, var_collections)
        self.scope = scope

    def _setup_graph(self):
        assert self.checkpoint_dir is not None, \
            "ModelSaver() doesn't have a valid checkpoint directory."
        vars = []
        for key in self.var_collections:
            vars.extend(tf.get_collection(key, scope=self.scope))
        vars = list(set(vars))
        self.path = os.path.join(self.checkpoint_dir, 'model')
        self.saver = tf.train.Saver(
            var_list=vars,
            max_to_keep=self._max_to_keep,
            keep_checkpoint_every_n_hours=self._keep_every_n_hours,
            write_version=tf.train.SaverDef.V2,
            save_relative_paths=True)
        # Scaffold will call saver.build from this collection
        tf.add_to_collection(tf.GraphKeys.SAVERS, self.saver)


def get_dataflow(list_filename, batch_size):
    ds = PNGDataFlow(imagedir=FLAGS.img_dir, imagelistfile=list_filename,
                     gtfile=FLAGS.ground_truth_file, have_imgname=False, shuffle=True,
                     img_num=FLAGS.img_num, batch_size=batch_size)
    parallel = min(40, multiprocessing.cpu_count() // 2)
    ds = PrefetchDataZMQ(ds, parallel)
    ds = BatchData(ds, batch_size, remainder=False)
    return ds


def get_config(model):
    nr_tower = max(get_num_gpu(), 1)
    assert FLAGS.batch % nr_tower == 0
    batch = FLAGS.batch // nr_tower

    logger.info("Running on {} towers. Batch size per tower: {}".format(nr_tower, batch))

    data = QueueInput(get_dataflow(FLAGS.train_list_filename, batch))

    # learning rate
    START_LR = FLAGS.lr
    BASE_LR = START_LR * (FLAGS.batch / 256.0)
    lr_list = []
    for idx, decay_point in enumerate(FLAGS.lr_decay_points):
        lr_list.append((decay_point, BASE_LR * 0.1 ** idx))
    callbacks = [
        ScopeModelSaver(checkpoint_dir=FLAGS.RHP_savepath, scope='RHP'),
        EstimatedTimeLeft(),
        ScheduledHyperParamSetter('learning_rate', lr_list),
    ]

    if get_num_gpu() > 0:
        callbacks.append(GPUUtilizationTracker())

    return TrainConfig(
        model=model,
        data=data,
        callbacks=callbacks,
        steps_per_epoch=FLAGS.steps_per_epoch // FLAGS.batch,
        max_epoch=FLAGS.max_epoch,
        session_init=MultipleRestore()
    )


if __name__ == '__main__':
    model = Model()
    logger.set_logger_dir(os.path.join('../log/', FLAGS.result_dir.split('/')[-1]))

    config = get_config(model)
    trainer = SyncMultiGPUTrainerReplicated(max(get_num_gpu(), 1))
    launch_train_with_config(config, trainer)
    print("Train finished, you may consider run these two lines for attack and universal attack\n")
    import sys

    print("python attack.py " + " ".join(sys.argv[1:]) + "\n")
    print("python attack.py " + " ".join(sys.argv[1:]) + " --universal" + "\n")
