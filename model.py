import tensorflow as tf

from preprocessing import lenet_preprocessing
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v2


def resnet101(images, classes=10):
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits, end_points = resnet_v2.resnet_v2_101(images, classes, is_training=True)

    # net = slim.conv2d(images, 20, [5, 5], scope='conv1')
    # net = slim.max_pool2d(net, [2, 2], scope='pool1')
    # net = slim.conv2d(net, 50, [5, 5], scope='conv2')
    # net = slim.max_pool2d(net, [2, 2], scope='pool2')
    # net = slim.flatten(net, scope='flatten3')
    # net = slim.fully_connected(net, 500, scope='fc4')
    # net = slim.fully_connected(net, 10, activation_fn=None, scope='fc5')

    logits = logits[:, 0, 0, :]

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
