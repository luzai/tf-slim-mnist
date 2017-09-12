import tensorflow as tf

from datasets import cifar100
from model import resnet101, load_batch
import utils
import numpy as np
import tensorflow.contrib.slim as slim

utils.init_dev(utils.get_dev(ok=[0, 1]))

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/tmp/cifar100',
                    'Directory with the mnist data.')
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
# flags.DEFINE_integer('num_batches', 1,
#                      'Num of batches to train (epochs).')
flags.DEFINE_string('log_dir',
                    # './output/cifar100-4',
                    'dbg',
                    'Directory with the log data.')
FLAGS = flags.FLAGS
alpha = 1.
beta = 0.
gamma = 0.


def get_loss(predictions, labels, classes):
    one_hot_labels = slim.one_hot_encoding(labels, classes)
    loss = slim.losses.softmax_cross_entropy(logits=predictions, onehot_labels=one_hot_labels)
    return loss


def get_acc(predictions, labels, name):
    with tf.name_scope(name):
        acc = slim.metrics.accuracy(tf.to_int64(tf.argmax(predictions, 1)), labels)
    return acc


def construct_loss(predictions, labels, name):
    with tf.name_scope(name):
        loss_100 = get_loss(predictions, labels, 100)

        # predictions_reshape = tf.reshape(tf.nn.softmax(predictions), (-1, 10, 10))
        # loss_10 = tf.multiply(tf.constant(beta),
        #                       slim.losses.log_loss(
        #                           tf.reduce_sum(predictions_reshape, axis=-1), one_hot_labels)
        #                       )
        #
        # for ind in range(10):
        #     predictions_ = predictions_reshape[ind, :]

        loss_reg = tf.reduce_sum(slim.losses.get_regularization_losses())

        total_loss = loss_100 + loss_reg

        slim.summary.scalar('loss/reg', loss_reg)
        slim.summary.scalar('loss/ttl', total_loss)
        slim.summary.scalar('loss/100', loss_100)

    return total_loss, [loss_reg, loss_100]


def main(args):
    tf.logging.set_verbosity(tf.logging.DEBUG)

    # load batch of dataset
    images, labels = load_batch(
        cifar100.get_split('train', FLAGS.data_dir),
        FLAGS.batch_size,
        is_training=True)
    images_val, labels_val = load_batch(cifar100.get_split('test', FLAGS.data_dir), FLAGS.batch_size, is_training=False)

    with tf.variable_scope('', reuse=False):
        predictions = resnet101(images, classes=100)

    total_loss, loss_list = construct_loss(predictions, labels, name='train')
    acc_train = get_acc(predictions, labels, 'val')

    global_step = slim.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(1e-1, global_step,
                                               1000, 0.5, staircase=True)
    tf.summary.scalar('lr', learning_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    print 'all loss ', slim.losses.get_losses()
    # create train op
    train_op = slim.learning.create_train_op(
        total_loss,
        optimizer,
        summarize_gradients=False,
        variables_to_train=slim.get_variables('resnet_v2_101/block4/.*/weights') + slim.get_variables(
            'resnet_v2_101/logits/*') + slim.get_variables('resnet_v2_101/block4/.*/beta') + slim.get_variables(
            'resnet_v2_101/block4/.*/gamma'),
        # slim.get_variables('resnet_v2_101/logits'),
    )

    variables_to_restore = slim.get_variables_to_restore(exclude=["resnet_v2_101/logits", ".*Ftrl.*", ".*Momentum.*"])

    checkpoint_path = './models/resnet_v2_101.ckpt'
    # checkpoint_path = './output/cifar100/model.ckpt-5564631'

    init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
        checkpoint_path, variables_to_restore)

    global_step = slim.get_or_create_global_step()
    global_step_init = tf.assign(global_step, 0)

    def InitAssignFn(sess):
        print 'init from pretrained model'
        sess.run([init_assign_op, global_step_init], init_feed_dict)
        sess.run(global_step_init)

    with tf.variable_scope('', reuse=True):
        predictions_val = resnet101(images_val, classes=100)

    acc_val = get_acc(predictions_val, labels_val, 'val')
    loss_val, loss_val_list = construct_loss(predictions_val, labels_val, 'val')

    def train_step_fn(session, *args, **kwargs):
        from tensorflow.contrib.slim.python.slim.learning import train_step

        total_loss, should_stop = train_step(session, *args, **kwargs)
        print 'diff ', session.run(train_step_fn.loss), total_loss
        if train_step_fn.step % 10 == 0:
            acc_ = session.run(train_step_fn.acc_train)

            acc_val_l_ = []
            for ind in range(10000 / FLAGS.batch_size + 1):
                acc_val_l_.append(session.run(train_step_fn.acc_val))
            acc_val_ = np.mean(acc_val_l_)

            print('Step %s - Loss: %.2f acc_val: %.2f%% acc: %.2f%%' % (
                str(train_step_fn.step).rjust(6, '0'), total_loss,
                acc_val_ * 100, acc_ * 100))

        train_step_fn.step += 1
        return [total_loss, should_stop]

    train_step_fn.step = 0
    train_step_fn.loss_val = loss_val
    train_step_fn.loss_val_list = loss_val_list
    train_step_fn.acc_val = acc_val
    train_step_fn.acc_train = acc_train
    train_step_fn.loss = total_loss
    train_step_fn.loss_list = loss_list

    # run training
    _sess_config = tf.ConfigProto(allow_soft_placement=True)
    _sess_config.gpu_options.allow_growth = True
    slim.learning.train(
        train_op,
        FLAGS.log_dir,
        init_fn=InitAssignFn,
        save_summaries_secs=10,
        session_config=_sess_config,
        number_of_steps=5000,
        log_every_n_steps=10,
        train_step_fn=train_step_fn,
        # trace_every_n_steps=20,
    )


if __name__ == '__main__':
    utils.rm(FLAGS.log_dir)
    tf.app.run()
