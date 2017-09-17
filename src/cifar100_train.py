import tensorflow as tf

from datasets import cifar100
from datasets.cifar100 import load_batch
from model import resnet101, resnet50
import utils, numpy as np
import tensorflow.contrib.slim as slim

utils.init_dev(utils.get_dev(ok=[0, 1, 2, 3]))
from hypers import cifar100 as FLAGS


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

    # run the image through the model
    predictions, end_points = resnet50(images, classes=100)
    # todo add svd observation

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
        labels=one_hot_labels_coarse, weights=FLAGS.beta
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
            weights=FLAGS.gamma)
        if ind <= 5:
            tf.summary.scalar('loss/group/group{}/train'.format(ind), loss_group_)
        loss_group_l.append(loss_group_)

    loss_group = tf.add_n(loss_group_l)
    if FLAGS.multi_loss:
        tf.losses.add_loss(loss_group)

    print '>> loss', tf.losses.get_losses(), len(tf.losses.get_losses())

    loss_reg = tf.add_n(tf.losses.get_regularization_losses())
    total_loss = tf.losses.get_total_loss()
    # ema = tf.train.ExponentialMovingAverage(decay=0.9)
    # total_loss_avg_op = ema.apply([total_loss])
    # total_loss_avg = ema.average(total_loss)
    slim.summary.scalar('loss/total/train', total_loss)
    # slim.summary.scalar('loss/total_avg', total_loss_avg)
    slim.summary.scalar('loss/loss100/train', loss_100)
    slim.summary.scalar('loss/reg/train', loss_reg)
    slim.summary.scalar('loss/loss20/train', loss_20)
    slim.summary.scalar('loss/group/total/train', loss_group)

    global_step = slim.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(
        FLAGS.init_lr, global_step,
        FLAGS.lr_decay_per_steps, FLAGS.lr_decay, staircase=True)
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
    # with tf.control_dependencies([total_loss]):
    #     train_op = tf.group(train_op, total_loss_avg_op)

    variables_to_restore = slim.get_variables_to_restore(
        exclude=[".*logits.*", ".*Ftrl.*", '.*Momentum.*', 'global_step'])

    init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
        FLAGS.checkpoint_path, variables_to_restore)

    def InitAssignFn(sess):
        print 'init from pretrained model'
        sess.run([init_assign_op, ], init_feed_dict)

    acc = slim.metrics.accuracy(tf.to_int64(tf.argmax(predictions, 1)), labels)

    slim.summary.scalar('acc/train', acc)

    def train_step_fn(session, *args, **kwargs):
        from tensorflow.contrib.slim.python.slim.learning import train_step

        total_loss_, should_stop = train_step(session, *args, **kwargs)
        # total_loss_ = 0
        if train_step_fn.step % 196 == 0:
            acc_ = session.run(train_step_fn.acc)
            print acc_
            print('>> Step {} - Loss: {} acc: {}%'.format(
                str(train_step_fn.step).rjust(6, '0'), total_loss_,
                np.mean(acc_) * 100))

        train_step_fn.step += 1
        return [total_loss_, should_stop]

    train_step_fn.step = 0
    train_step_fn.acc = acc

    # run training

    _sess_config = tf.ConfigProto(allow_soft_placement=True)
    _sess_config.gpu_options.allow_growth = True
    slim.learning.train(
        train_op,
        FLAGS.log_dir,
        init_fn=InitAssignFn,
        save_summaries_secs=50,
        save_interval_secs=50,
        session_config=_sess_config,
        number_of_steps=None,
        log_every_n_steps=196,
        train_step_fn=train_step_fn,
        # trace_every_n_steps=20,
    )


if __name__ == '__main__':
    utils.rm(FLAGS.log_dir)
    proc=utils.shell('python cifar100_eval.py', block=False)
    # print proc.communicate()[-1]
    tf.app.run()
