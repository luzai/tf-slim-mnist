import tensorflow as tf

from datasets import imagenet
from model import load_batch
# from model import resnet101_2 as resnet101
from  model import resnet101
import utils
import numpy as np
import tensorflow.contrib.slim as slim

utils.init_dev(utils.get_dev(ok=[0, 1, 2, 3]))

flags = tf.app.flags
flags.DEFINE_string('data_dir', '../Perf_Pred/data/imagenet10k',
                    'Directory with the mnist data.')
flags.DEFINE_integer('batch_size', 32, 'Batch size.')

flags.DEFINE_string('log_dir',
                    './output/imagenet101',
                    # 'dbg2',
                    'Directory with the log data.')
FLAGS = flags.FLAGS


def main(args):
    tf.logging.set_verbosity(tf.logging.DEBUG)
    # load the dataset
    dataset = imagenet.get_split('train', FLAGS.data_dir)

    # load batch of dataset
    images, labels = load_batch(
        dataset,
        FLAGS.batch_size,
        height=224,
        width=224,
        is_training=True)

    # run the image through the model

    predictions = resnet101(images, classes=7461)

    # print slim.get_variables()


    beta = 0.1
    gamma = 0.1

    # get the cross-entropy loss
    one_hot_labels = slim.one_hot_encoding(
        labels,
        dataset.num_classes)

    # todo hinge loss
    loss_100 = slim.losses.softmax_cross_entropy(
        logits=predictions,
        onehot_labels=one_hot_labels)

    # slim.losses.add_loss(loss_100)

    loss_reg = tf.reduce_sum(slim.losses.get_regularization_losses())

    print slim.losses.get_losses()
    print tf.get_collection(tf.GraphKeys.LOSSES)

    total_loss = slim.losses.get_total_loss()
    slim.summary.scalar('loss/total', total_loss)
    slim.summary.scalar('loss/loss100', loss_100)
    slim.summary.scalar('loss/loss_reg', loss_reg)

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
        variables_to_train=slim.get_trainable_variables('resnet_v2_101/block4/.*') + slim.get_trainable_variables(
            '.*logits.*'),
        # slim.get_variables('resnet_v2_101/logits'),
    )

    variables_to_restore = slim.get_variables_to_restore(
        exclude=[".*logits.*", ".*Ftrl.*", '.*Momentum.*', '.*fully_connected.*'])

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

    images_val, labels_val = load_batch(imagenet.get_split('validation', FLAGS.data_dir), height=224,
                                        width=224, batch_size=FLAGS.batch_size, is_training=False)

    with tf.variable_scope('', reuse=True):
        predictions_val = resnet101(images_val, classes=7461)

    acc_val = slim.metrics.accuracy(tf.to_int64(tf.argmax(predictions_val, 1)), labels_val)

    acc = slim.metrics.accuracy(tf.to_int64(tf.argmax(predictions, 1)), labels)

    slim.summary.scalar('acc/train', acc)
    slim.summary.scalar('acc/val', acc_val)

    def train_step_fn(session, *args, **kwargs):
        from tensorflow.contrib.slim.python.slim.learning import train_step

        total_loss, should_stop = train_step(session, *args, **kwargs)

        if train_step_fn.step % 20 == 0:
            acc_ = session.run(train_step_fn.acc)
            # acc_vall_ = []
            # for ind in range(10000 // FLAGS.batch_size + 1):
            #     acc_vall_.append(session.run(train_step_fn.acc_val))

            print('>> Step %s - Loss: %.2f  acc: %.2f%%' % (
                str(train_step_fn.step).rjust(6, '0'), total_loss,
                # np.mean(acc_vall_) * 100,
                acc_ * 100))

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
        save_interval_secs=600,
        save_summaries_secs=600,
        session_config=_sess_config,
        number_of_steps=None,
        log_every_n_steps=20,
        train_step_fn=train_step_fn,
        # trace_every_n_steps=20,
    )


if __name__ == '__main__':
    utils.rm(FLAGS.log_dir)
    tf.app.run()
