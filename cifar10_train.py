import tensorflow as tf

from datasets import cifar10
from model import resnet101, load_batch
import utils

import tensorflow.contrib.slim as slim

utils.init_dev(utils.get_dev(ok=[0, 1]))

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/tmp/cifar10',
                    'Directory with the mnist data.')
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
# flags.DEFINE_integer('num_batches', 1,
#                      'Num of batches to train (epochs).')
flags.DEFINE_string('log_dir', './log2',
                    'Directory with the log data.')
FLAGS = flags.FLAGS


def main(args):
    tf.logging.set_verbosity(tf.logging.DEBUG)
    # load the dataset
    dataset = cifar10.get_split('train', FLAGS.data_dir)

    # load batch of dataset
    images, labels = load_batch(
        dataset,
        FLAGS.batch_size,
        is_training=True)

    # run the image through the model

    predictions = resnet101(images)

    # get the cross-entropy loss
    one_hot_labels = slim.one_hot_encoding(
        labels,
        dataset.num_classes)
    slim.losses.softmax_cross_entropy(
        predictions,
        one_hot_labels)
    total_loss = slim.losses.get_total_loss()
    tf.summary.scalar('loss', total_loss)

    # use RMSProp to optimize
    optimizer = tf.train.MomentumOptimizer(0.001, 0.9)
    # optimizer = tf.train.RMSPropOptimizer(0.001, 0.9)
    # optimizer = tf.train.GradientDescentOptimizer(0.001)
    # create train op
    train_op = slim.learning.create_train_op(
        total_loss,
        optimizer,
        summarize_gradients=False,
        # variables_to_train=slim.get_variables('resnet_v2_101/logits'),
    )

    variables_to_restore = slim.get_variables_to_restore(exclude=["resnet_v2_101/logits"])

    checkpoint_path = './models/resnet_v2_101.ckpt'
    init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
        checkpoint_path, variables_to_restore)

    global_step = slim.get_or_create_global_step()
    global_step_init = tf.assign(global_step, 0)

    def InitAssignFn(sess):
        print 'init from pretrained model'
        sess.run([init_assign_op, global_step_init], init_feed_dict)

    images_val, labels_val = load_batch(cifar10.get_split('test', FLAGS.data_dir), FLAGS.batch_size, is_training=False)

    with tf.variable_scope('', reuse=True):
        predictions_val = resnet101(images_val)

    acc_val = slim.metrics.accuracy(tf.to_int64(tf.argmax(predictions_val, 1)), labels_val)

    acc = slim.metrics.accuracy(tf.to_int64(tf.argmax(predictions, 1)), labels)

    tf.summary.scalar('acc/train', acc)
    tf.summary.scalar('acc/val', acc_val)

    def train_step_fn(session, *args, **kwargs):
        from tensorflow.contrib.slim.python.slim.learning import train_step

        total_loss, should_stop = train_step(session, *args, **kwargs)

        if train_step_fn.step % 20 == 0:
            # acc_val_, acc_ = 0, 0
            acc_val_,acc_ = session.run([train_step_fn.acc_val,train_step_fn.acc])

            print('Step %s - Loss: %.2f acc_val: %.2f%% acc: %.2f%%' % (
                str(train_step_fn.step).rjust(6, '0'), total_loss,
                acc_val_ * 100, acc_ * 100))

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
        save_summaries_secs=60,
        session_config=_sess_config,
        number_of_steps=5562631 + 2000,
        log_every_n_steps=20,
        train_step_fn=train_step_fn,
        # trace_every_n_steps=20,
    )


if __name__ == '__main__':
    utils.rm(FLAGS.log_dir)
    tf.app.run()
