# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the Cifar10 dataset.

The dataset scripts used to create the dataset can be found at:
tensorflow/models/slim/datasets/download_and_convert_cifar10.py
"""

from __future__ import division

import os
import tensorflow as tf
import numpy as np
import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = 'cifar100_%s.tfrecord'

SPLITS_TO_SIZES = {'train': 50000, 'test': 10000}

_NUM_CLASSES = 100

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A [32 x 32 x 3] color image.',
    'label': 'A single integer between 0 and 99',
}

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


def unpickle(file_path):
    import cPickle
    with open(file_path, 'rb') as f:
        data = cPickle.load(f)
    return data


mapping = unpickle('../data/cifar100/mapping.pkl')
mapping_human = unpickle('../data/cifar100/mapping_human.pkl')

mapp = {
    "0": 4,
    "1": 1,
    "2": 14,
    "3": 8,
    "4": 0,
    "5": 6,
    "6": 7,
    "7": 7,
    "8": 18,
    "9": 3,
    "10": 3,
    "11": 14,
    "12": 9,
    "13": 18,
    "14": 7,
    "15": 11,
    "16": 3,
    "17": 9,
    "18": 7,
    "19": 11,
    "20": 6,
    "21": 11,
    "22": 5,
    "23": 10,
    "24": 7,
    "25": 6,
    "26": 13,
    "27": 15,
    "28": 3,
    "29": 15,
    "30": 0,
    "31": 11,
    "32": 1,
    "33": 10,
    "34": 12,
    "35": 14,
    "36": 16,
    "37": 9,
    "38": 11,
    "39": 5,
    "40": 5,
    "41": 19,
    "42": 8,
    "43": 8,
    "44": 15,
    "45": 13,
    "46": 14,
    "47": 17,
    "48": 18,
    "49": 10,
    "50": 16,
    "51": 4,
    "52": 17,
    "53": 4,
    "54": 2,
    "55": 0,
    "56": 17,
    "57": 4,
    "58": 18,
    "59": 17,
    "60": 10,
    "61": 3,
    "62": 2,
    "63": 12,
    "64": 12,
    "65": 16,
    "66": 12,
    "67": 1,
    "68": 9,
    "69": 19,
    "70": 2,
    "71": 10,
    "72": 0,
    "73": 1,
    "74": 16,
    "75": 12,
    "76": 9,
    "77": 13,
    "78": 15,
    "79": 13,
    "80": 16,
    "81": 19,
    "82": 2,
    "83": 4,
    "84": 6,
    "85": 19,
    "86": 5,
    "87": 5,
    "88": 8,
    "89": 19,
    "90": 18,
    "91": 1,
    "92": 2,
    "93": 15,
    "94": 6,
    "95": 0,
    "96": 17,
    "97": 8,
    "98": 14,
    "99": 13
}


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    """Gets a dataset tuple with instructions for reading cifar10.

    Args:
      split_name: A train/test split name.
      dataset_dir: The base directory of the dataset sources.
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
      reader: The TensorFlow reader type.

    Returns:
      A `Dataset` namedtuple.

    Raises:
      ValueError: if `split_name` is not a valid train/test split.
    """
    if split_name not in SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)

    if not file_pattern:
        file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    # Allowing None in the signature so that dataset_factory can use the default.
    if not reader:
        reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/class/label': tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(shape=[32, 32, 3]),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None
    if dataset_utils.has_labels(dataset_dir):
        labels_to_names = dataset_utils.read_label_file(dataset_dir)

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=SPLITS_TO_SIZES[split_name],
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=_NUM_CLASSES,
        labels_to_names=labels_to_names)