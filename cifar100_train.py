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
                    './output/cifar100-non-freeze',
                    # 'dbg2',
                    'Directory with the log data.')
FLAGS = flags.FLAGS


def map_label(input_tensor):
    keys = np.array(cifar100.mapp.keys(), dtype=np.int64)
    values = np.array(cifar100.mapp.values(), dtype=np.int64)
    table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(keys, values), -1)
    out = table.lookup(input_tensor)
    return out


def main(args):
    tf.logging.set_verbosity(tf.logging.DEBUG)
    # load the dataset
    dataset = cifar100.get_split('train', FLAGS.data_dir)

    # load batch of dataset
    images, labels = load_batch(
        dataset,
        FLAGS.batch_size,
        is_training=True)

    # run the image through the model

    predictions = resnet101(images, classes=100)

    alpha = 1.
    beta = 0.1
    gamma = 0.1

    # get the cross-entropy loss
    one_hot_labels = slim.one_hot_encoding(
        labels,
        dataset.num_classes)

    loss_100 = alpha * slim.losses.softmax_cross_entropy(
        predictions,
        one_hot_labels)

    labels_coarse = map_label(labels)
    one_hot_labels_coarse = slim.one_hot_encoding(labels_coarse, 20)

    predictions_reshape = tf.reshape(tf.nn.softmax(predictions), (-1, 20, 5))
    loss_20 = beta * slim.losses.log_loss(
        tf.reduce_sum(predictions_reshape, axis=-1), one_hot_labels_coarse)

    # slim.losses.add_loss(loss_20)
    # for ind in range(10):
    #     predictions_ = predictions_reshape[ind, :]

    loss_reg = tf.reduce_sum(slim.losses.get_regularization_losses())
    total_loss = slim.losses.get_total_loss()
    slim.summary.scalar('loss/total', total_loss)
    slim.summary.scalar('loss/loss100', loss_100)
    slim.summary.scalar('loss/loss_reg', loss_reg)
    # slim.summary.scalar('loss/loss10', loss_10)


    global_step = slim.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(1e-1, global_step,
                                               1000, 0.5, staircase=True)
    slim.summary.scalar('lr', learning_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)

    # create train op
    train_op = slim.learning.create_train_op(
        total_loss,
        optimizer,
        summarize_gradients=False,
        # variables_to_train=slim.get_variables('resnet_v2_101/block4/.*/weights') + slim.get_variables(
        #     'resnet_v2_101/logits/*') + slim.get_variables('resnet_v2_101/block4/.*/beta') + slim.get_variables(
        #     'resnet_v2_101/block4/.*/gamma'),
        # slim.get_variables('resnet_v2_101/logits'),
    )

    variables_to_restore = slim.get_variables_to_restore(exclude=["resnet_v2_101/logits", ".*Ftrl.*", '.*Momentum.*'])

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

    images_val, labels_val = load_batch(cifar100.get_split('test', FLAGS.data_dir), FLAGS.batch_size, is_training=False)

    with tf.variable_scope('', reuse=True):
        predictions_val = resnet101(images_val, classes=100)

    acc_val = slim.metrics.accuracy(tf.to_int64(tf.argmax(predictions_val, 1)), labels_val)

    acc = slim.metrics.accuracy(tf.to_int64(tf.argmax(predictions, 1)), labels)

    slim.summary.scalar('acc/train', acc)
    slim.summary.scalar('acc/val', acc_val)

    def train_step_fn(session, *args, **kwargs):
        from tensorflow.contrib.slim.python.slim.learning import train_step

        total_loss, should_stop = train_step(session, *args, **kwargs)

        if train_step_fn.step % 196 == 0:
            acc_ = session.run(train_step_fn.acc)
            acc_vall_ = []
            for ind in range(10000 // FLAGS.batch_size + 1):
                acc_vall_.append(session.run(train_step_fn.acc_val))

            print('>> Step %s - Loss: %.2f acc_val: %.2f%% acc: %.2f%%' % (
                str(train_step_fn.step).rjust(6, '0'), total_loss,
                np.mean(acc_vall_) * 100, acc_ * 100))

        train_step_fn.step += 1
        return [total_loss, should_stop]

    train_step_fn.step = 0
    train_step_fn.acc_val = acc_val
    train_step_fn.acc = acc

    # run training

    _sess_config = tf.ConfigProto(allow_soft_placement=True)
    _sess_config.gpu_options.allow_growth = True
    slim.learning.train(
        train_op,
        FLAGS.log_dir,
        init_fn=InitAssignFn,
        save_summaries_secs=50,
        session_config=_sess_config,
        number_of_steps=8000,
        log_every_n_steps=196,
        train_step_fn=train_step_fn,
        # trace_every_n_steps=20,
    )


if __name__ == '__main__':
    utils.rm(FLAGS.log_dir)
    tf.app.run()
