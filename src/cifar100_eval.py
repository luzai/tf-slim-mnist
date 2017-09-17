import tensorflow as tf

from datasets import cifar100
from model import resnet101, resnet50
from datasets.cifar100 import load_batch

import tensorflow.contrib.slim as slim

import utils

utils.init_dev(utils.get_dev())
from hypers import cifar100_eval as FLAGS


def main(args):
    # load the dataset
    dataset = cifar100.get_split('test', FLAGS.data_dir, )

    # load batch
    batch_queue = load_batch(
        dataset,
        FLAGS.batch_size,
        is_training=False)
    images, labels = batch_queue.dequeue()
    # get the model prediction
    predictions, end_points = resnet50(images, classes=100)

    # convert prediction values for each class into single class prediction

    one_hot_labels = slim.one_hot_encoding(
        labels,
        dataset.num_classes)
    tf.losses.softmax_cross_entropy(
        logits=predictions,
        onehot_labels=one_hot_labels)

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
    metric_map = {}
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
            metric_map['loss/group/group{}/val'.format(ind)] = slim.metrics.streaming_mean(loss_group_)
        loss_group_l.append(loss_group_)

    loss_group = tf.add_n(loss_group_l)
    if FLAGS.multi_loss:
        tf.losses.add_loss(loss_group)

    print '>> loss', tf.losses.get_losses(), len(tf.losses.get_losses())

    loss_reg = tf.add_n(tf.losses.get_regularization_losses())

    total_loss = tf.losses.get_total_loss()
    # tf.summary.scalar('loss/val/ori', total_loss)

    # streaming metrics to evaluate
    predictions = tf.to_int64(tf.argmax(predictions, 1))
    metrics_to_values, metrics_to_updates = slim.metrics.aggregate_metric_map(
        utils.dict_concat([{
            # 'mse/val': slim.metrics.streaming_mean_squared_error(predictions, labels),
            'acc/val': slim.metrics.streaming_accuracy(predictions, labels),
            'loss/ttl/val': slim.metrics.streaming_mean(total_loss),
            'loss/100/val': slim.metrics.streaming_mean(loss_100),
            'loss/20/val': slim.metrics.streaming_mean(loss_20),
            'loss/reg/val': slim.metrics.streaming_mean(loss_reg),
            'loss/group/total/train': slim.metrics.streaming_mean(loss_group)
        }, metric_map]))

    # write the metrics as summaries
    for metric_name, metric_value in metrics_to_values.iteritems():
        tf.summary.scalar(metric_name + '/values', metric_value)
    # for metric_name, metric_value in metrics_to_updates.iteritems():
    #     tf.summary.scalar(metric_name+'/update', metric_value)

    # evaluate on the model saved at the checkpoint directory
    # evaluate every eval_interval_secs
    tf.logging.set_verbosity(tf.logging.DEBUG)
    _sess_config = tf.ConfigProto(allow_soft_placement=True)
    _sess_config.gpu_options.allow_growth = True
    slim.evaluation.evaluation_loop(
        '',
        FLAGS.checkpoint_dir,
        FLAGS.log_dir,
        num_evals=FLAGS.num_evals,
        eval_op=metrics_to_updates.values(),
        session_config=_sess_config,
        eval_interval_secs=FLAGS.eval_interval_secs)


if __name__ == '__main__':
    utils.rm(FLAGS.log_dir)
    tf.app.run()
