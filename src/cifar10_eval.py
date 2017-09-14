import tensorflow as tf

from datasets import cifar10
from model import resnet101, load_batch

import tensorflow.contrib.slim as slim

metrics = tf.contrib.metrics

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/tmp/cifar10',
                    'Directory with the MNIST data.')
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_integer('eval_interval_secs', 60,
                     'Number of seconds between evaluations.')
flags.DEFINE_integer('num_evals', 50, 'Number of batches to evaluate.')
flags.DEFINE_string('log_dir', './log/eval',
                    'Directory where to log evaluation data.')
flags.DEFINE_string('checkpoint_dir', './log/train',
                    'Directory with the model checkpoint data.')
FLAGS = flags.FLAGS


def main(args):
    # load the dataset
    dataset = cifar10.get_split('test', FLAGS.data_dir)

    # load batch
    images, labels = load_batch(
        dataset,
        FLAGS.batch_size,
        is_training=False)

    # get the model prediction
    predictions = resnet101(images)

    # convert prediction values for each class into single class prediction
    predictions = tf.to_int64(tf.argmax(predictions, 1))

    # streaming metrics to evaluate
    metrics_to_values, metrics_to_updates = metrics.aggregate_metric_map({
        'mse': metrics.streaming_mean_squared_error(predictions, labels),
        'accuracy': metrics.streaming_accuracy(predictions, labels),
    })

    # write the metrics as summaries
    for metric_name, metric_value in metrics_to_values.iteritems():
        tf.summary.scalar(metric_name, metric_value)

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
    tf.app.run()
