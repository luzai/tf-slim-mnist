# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Downloads and converts cifar10 data to TFRecords of TF-Example protos.

This module downloads the cifar10 data, uncompresses it, reads the files
that make up the cifar10 data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take several minutes to run.

"""

from __future__ import division

import os
import sys
import tarfile

import numpy as np
from six.moves import cPickle
from six.moves import urllib
import tensorflow as tf

# from datasets import dataset_utils
import dataset_utils

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset_dir', '/tmp/cifar100', 'cifar10 data dir')

# The URL where the CIFAR data can be downloaded.
_DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'

# The height and width of each image.
_IMAGE_SIZE = 32

# The names of the classes.

coarse_labels = ['aquatic_mammals',
                 'fish',
                 'flowers',
                 'food_containers',
                 'fruit_and_vegetables',
                 'household_electrical_devices',
                 'household_furniture',
                 'insects',
                 'large_carnivores',
                 'large_man-made_outdoor_things',
                 'large_natural_outdoor_scenes',
                 'large_omnivores_and_herbivores',
                 'medium_mammals',
                 'non-insect_invertebrates',
                 'people',
                 'reptiles',
                 'small_mammals',
                 'trees',
                 'vehicles_1',
                 'vehicles_2']
coarse_labels_human = np.array(coarse_labels)
fine_labels = [['apple', 'aquarium_fish', 'baby', 'bear', 'beaver'],
               ['bed', 'bee', 'beetle', 'bicycle', 'bottle'],
               ['bowl', 'boy', 'bridge', 'bus', 'butterfly'],
               ['camel', 'can', 'castle', 'caterpillar', 'cattle'],
               ['chair', 'chimpanzee', 'clock', 'cloud', 'cockroach'],
               ['couch', 'crab', 'crocodile', 'cup', 'dinosaur'],
               ['dolphin', 'elephant', 'flatfish', 'forest', 'fox'],
               ['girl', 'hamster', 'house', 'kangaroo', 'keyboard'],
               ['lamp', 'lawn_mower', 'leopard', 'lion', 'lizard'],
               ['lobster', 'man', 'maple_tree', 'motorcycle', 'mountain'],
               ['mouse', 'mushroom', 'oak_tree', 'orange', 'orchid'],
               ['otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree'],
               ['plain', 'plate', 'poppy', 'porcupine', 'possum'],
               ['rabbit', 'raccoon', 'ray', 'road', 'rocket'],
               ['rose', 'sea', 'seal', 'shark', 'shrew'],
               ['skunk', 'skyscraper', 'snail', 'snake', 'spider'],
               ['squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table'],
               ['tank', 'telephone', 'television', 'tiger', 'tractor'],
               ['train', 'trout', 'tulip', 'turtle', 'wardrobe'],
               ['whale', 'willow_tree', 'wolf', 'woman', 'worm']]

fine_labels_human = np.array(fine_labels).flatten()


def _add_to_tfrecord(filename, tfrecord_writer, offset=0):
    """Loads data from the cifar10 pickle files and writes files to a TFRecord.

    Args:
      filename: The filename of the cifar10 pickle file.
      tfrecord_writer: The TFRecord writer to use for writing.
      offset: An offset into the absolute number of images previously written.

    Returns:
      The new offset.
    """
    with tf.gfile.Open(filename, 'rb') as f:
        if sys.version_info < (3,):
            data = cPickle.load(f)
        else:
            data = cPickle.load(f, encoding='bytes')

    images = data[b'data']
    num_images = images.shape[0]

    images = images.reshape((num_images, 3, 32, 32))
    labels = data[b'fine_labels']
    coarse_labels = data[b'coarse_labels']
    from cifar100 import mapp
    labels = [mapp[l] for l in labels]
    mapping = {}
    mapping_human = {}

    for label, cl in zip(labels, coarse_labels):
        if cl not in mapping:
            mapping[cl] = {label}
            mapping_human[coarse_labels_human[cl]] = {fine_labels_human[label]}
        else:
            mapping[cl].add(label)
            mapping_human[coarse_labels_human[cl]].add(fine_labels_human[label])

    with tf.Graph().as_default():
        image_placeholder = tf.placeholder(dtype=tf.uint8)
        encoded_image = tf.image.encode_png(image_placeholder)

        with tf.Session('') as sess:
            for j in range(num_images):
                sys.stdout.write('\r>> Reading file [%s] image %d/%d' % (
                    filename, offset + j + 1, offset + num_images))
                sys.stdout.flush()

                image = np.squeeze(images[j]).transpose((1, 2, 0))
                label = labels[j]

                png_string = sess.run(encoded_image,
                                      feed_dict={image_placeholder: image})

                example = dataset_utils.image_to_tfexample(
                    png_string, b'png', _IMAGE_SIZE, _IMAGE_SIZE, label)
                tfrecord_writer.write(example.SerializeToString())

    def pickle(data, file_path):
        with open(file_path, 'wb') as f:
            cPickle.dump(data, f, cPickle.HIGHEST_PROTOCOL)

    # pickle(mapping, '/tmp/cifar100/mapping.pkl')
    # pickle(mapping_human, '/tmp/cifar100/mapping_human.pkl')

    return offset + num_images


def _get_output_filename(dataset_dir, split_name):
    """Creates the output filename.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      split_name: The name of the train/test split.

    Returns:
      An absolute file path.
    """
    return '%s/cifar100_%s.tfrecord' % (dataset_dir, split_name)


def _clean_up_temporary_files(dataset_dir):
    """Removes temporary files used to create the dataset.

    Args:
      dataset_dir: The directory where the temporary files are stored.
    """
    filename = _DATA_URL.split('/')[-1]
    filepath = os.path.join(dataset_dir, filename)
    tf.gfile.Remove(filepath)

    tmp_dir = os.path.join(dataset_dir, 'cifar-10-batches-py')
    tf.gfile.DeleteRecursively(tmp_dir)


def run(args):
    """Runs the download and conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
    """
    dataset_dir = FLAGS.dataset_dir

    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    training_filename = _get_output_filename(dataset_dir, 'train')
    testing_filename = _get_output_filename(dataset_dir, 'test')

    # if tf.gfile.Exists(training_filename) and tf.gfile.Exists(testing_filename):
    #     print('Dataset files already exist. Exiting without re-creating them.')
    #     return

    dataset_utils.download_and_uncompress_tarball(_DATA_URL, dataset_dir)

    # First, process the training data:
    with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
        offset = 0

        filename = os.path.join(dataset_dir,
                                'cifar-100-python', 'train')  # 1-indexed.
        offset = _add_to_tfrecord(filename, tfrecord_writer, offset)

    # Next, process the testing data:
    with tf.python_io.TFRecordWriter(testing_filename) as tfrecord_writer:
        filename = os.path.join(dataset_dir,
                                'cifar-100-python',
                                'test')
        _add_to_tfrecord(filename, tfrecord_writer)

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(fine_labels_human)), fine_labels_human))
    dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

    labels_to_class_names = dict(zip(range(len(coarse_labels_human)), coarse_labels_human))
    dataset_utils.write_label_file(labels_to_class_names, dataset_dir, filename='labels-coarse.txt')

    # _clean_up_temporary_files(dataset_dir)
    print('\nFinished converting the Cifar10 dataset!')


if __name__ == '__main__':
    tf.app.run(main=run)
