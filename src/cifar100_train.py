import tensorflow as tf

from datasets import cifar100
from datasets.cifar100 import load_batch
from model import resnet101, resnet50
import utils
import numpy as np
import tensorflow.contrib.slim as slim

utils.init_dev(utils.get_dev(ok=[0, 1, 2, 3]))

class DictClass(dict):
    __getattr__, __setattr__ = dict.get, dict.__setitem__


FLAGS = DictClass(dict(data_dir='../data/cifar100',
                       batch_size=128,
                       log_dir='../output/multiloss-dry3',
                       checkpoint_path='../models/resnet50/resnet_v2_50.ckpt',
                       multiloss=False,
                       ))


# flags.DEFINE_integer('num_batches', 1,
#                      'Num of batches to train (epochs).')


# def map_label(input_tensor):
#     keys = np.array(cifar100.mapp.keys(), dtype=np.int64)
#     values = np.array(cifar100.mapp.values(), dtype=np.int64)
#     table = tf.contrib.lookup.HashTable(
#         tf.contrib.lookup.KeyValueTensorInitializer(keys, values), -1)
#     out = table.lookup(input_tensor)
#     return out

def main(args):
    # train_op, InitAssignFn, train_step_fn = config_graph()

    tf.logging.set_verbosity(tf.logging.DEBUG)
    # load the dataset
    dataset = cifar100.get_split('train', FLAGS.data_dir)

    # load batch of dataset
    batch_queue = load_batch(
        dataset,
        FLAGS.batch_size,
        is_training=True)
    images, labels = batch_queue.dequeue()
    slim.summary.image('input/image', images)
    # slim.summary.image('input/label',labels)
    # run the image through the model
    predictions, end_points = resnet50(images, classes=100)

    beta = 1.
    gamma = 1.

    # get the cross-entropy loss
    one_hot_labels = slim.one_hot_encoding(
        labels,
        dataset.num_classes)

    loss_100 = tf.losses.softmax_cross_entropy(
        logits=predictions,
        onehot_labels=one_hot_labels)

    # labels_coarse = map_label(labels)
    labels_coarse = tf.to_int64(labels // 5)
    one_hot_labels_coarse = slim.one_hot_encoding(labels_coarse, 20)

    predictions_reshape = tf.reshape(tf.nn.softmax(predictions), (-1, 20, 5))
    loss_20 = tf.losses.log_loss(
        predictions=tf.reduce_sum(predictions_reshape, axis=-1),
        labels=one_hot_labels_coarse, weights=beta
        , loss_collection=None if not FLAGS.multi_loss else tf.GraphKeys.LOSSES
    )

    loss_group_l = []
    for ind in range(20):
        predictions_ = tf.reshape(predictions, (-1, 20, 5))
        bs = tf.shape(predictions_, out_type=tf.int64)[0]
        sel = tf.stack([tf.range(bs, dtype=tf.int64), labels // 5], axis=1)
        predictions_ = tf.gather_nd(predictions_, sel)

        one_hot_labels_group = slim.one_hot_encoding(tf.mod(labels, 5), 5)
        loss_group_ = tf.losses.softmax_cross_entropy(
            logits=predictions_,
            onehot_labels=one_hot_labels_group,
            loss_collection=None,
            weights=gamma)
        if ind<=5:
            tf.summary.scalar('loss/group/group'+str(ind), loss_group_)
        loss_group_l.append(loss_group_)

    loss_group = tf.add_n(loss_group_l)
    if FLAGS.multi_loss:
        tf.losses.add_loss(loss_group)

    print '>> loss', tf.losses.get_losses(), len(tf.losses.get_losses())  # tf.get_collection(tf.GraphKeys.LOSSES)

    loss_reg = tf.add_n(tf.losses.get_regularization_losses())
    total_loss = tf.losses.get_total_loss()
    slim.summary.scalar('loss/total', total_loss)
    slim.summary.scalar('loss/loss100', loss_100)
    slim.summary.scalar('loss/reg', loss_reg)
    slim.summary.scalar('loss/loss20', loss_20)
    slim.summary.scalar('loss/group', loss_group)

    global_step = slim.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(1e-1, global_step,
                                               1000, 0.5, staircase=True)
    slim.summary.scalar('lr', learning_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)

    # create train op
    train_op = slim.learning.create_train_op(
        total_loss,
        optimizer,
        # summarize_gradients=True,
        # variables_to_train=slim.get_variables('resnet_v2_101/block4/.*/weights') + slim.get_variables(
        #     'resnet_v2_101/logits/*') + slim.get_variables('resnet_v2_101/block4/.*/beta') + slim.get_variables(
        #     'resnet_v2_101/block4/.*/gamma'),
        # slim.get_variables('resnet_v2_101/logits'),
    )

    variables_to_restore = slim.get_variables_to_restore(
        exclude=[".*logits.*", ".*Ftrl.*", '.*Momentum.*', 'global_step'])

    init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
        FLAGS.checkpoint_path, variables_to_restore)

    global_step = slim.get_or_create_global_step()
    global_step_init = tf.assign(global_step, 0)

    def InitAssignFn(sess):
        print 'init from pretrained model'
        sess.run([init_assign_op, global_step_init], init_feed_dict)
        sess.run(global_step_init)

    batch_queue_val = load_batch(cifar100.get_split('test', FLAGS.data_dir), FLAGS.batch_size, is_training=False)
    images_val, labels_val = batch_queue_val.dequeue()
    with tf.variable_scope('', reuse=True):
        predictions_val, _ = resnet50(images_val, classes=100)

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
