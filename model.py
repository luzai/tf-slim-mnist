import tensorflow as tf

from preprocessing import lenet_preprocessing
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v2


def resnet101(images, classes=10):
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits, end_points = resnet_v2.resnet_v2_101(images, classes, is_training=True)

        logits = logits[:, 0, 0, :]

    return logits


import tensorflow.contrib.layers as layers_lib
from tensorflow.contrib.layers.python.layers import layers


def resnet101_2(images, classes):
    # tf.reset_default_graph()
    # with tf.variable_scope('',reuse=True):
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        # tf.get_variable_scope().reuse_variables()
        logits, end_points = resnet_v2.resnet_v2_101(images, global_pool=False, is_training=True)
        # logits = layers.batch_norm(
        #     logits, activation_fn=tf.nn.relu, scope='postnorm2')
        # logits = layers_lib.conv2d(logits, classes, [1, 1], padding='valid', scope='logits',
        #                            normalizer_fn=None,
        #                            activation_fn=None)
        # logits = tf.reduce_mean(logits, axis=1)
        # logits = tf.reduce_mean(logits, axis=1)
        logits = slim.max_pool2d(logits, (3, 3), stride=4)
        logits = slim.flatten(logits)
        logits = slim.fully_connected(logits, classes, scope='logits', activation_fn=None, normalizer_fn=None)
    return logits


def load_batch(dataset, batch_size, height=32, width=32, is_training=False):
    data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset)

    image, label = data_provider.get(['image', 'label'])

    image = lenet_preprocessing.preprocess_image(
        image,
        height,
        width,
        is_training)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        allow_smaller_final_batch=True)

    return images, labels
