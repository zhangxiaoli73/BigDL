#
# Licensed to Intel Corporation under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# Intel Corporation licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Part of the code originally from Tensorflow


import gzip

import numpy

import base

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

TRAIN_MEAN = 0.13066047740239506
TRAIN_STD = 0.3081078
TEST_MEAN = 0.13251460696903547
TEST_STD = 0.31048024


def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth].

    Args:
      f: A file object that can be passed into a gzip reader.

    Returns:
      data: A 4D unit8 numpy array [index, y, x, depth].

    Raises:
      ValueError: If the bytestream does not start with 2051.

    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def extract_labels(f):
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' %
                (magic, f.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        return labels


def read_data_sets(train_dir, data_type="train"):
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

    if data_type == "train":
        local_file = base.maybe_download(TRAIN_IMAGES, train_dir,
                                         SOURCE_URL + TRAIN_IMAGES)
        with open(local_file, 'rb') as f:
            train_images = extract_images(f)

        local_file = base.maybe_download(TRAIN_LABELS, train_dir,
                                         SOURCE_URL + TRAIN_LABELS)
        with open(local_file, 'rb') as f:
            train_labels = extract_labels(f)
        return train_images, train_labels

    else:
        local_file = base.maybe_download(TEST_IMAGES, train_dir,
                                         SOURCE_URL + TEST_IMAGES)
        with open(local_file, 'rb') as f:
            test_images = extract_images(f)

        local_file = base.maybe_download(TEST_LABELS, train_dir,
                                         SOURCE_URL + TEST_LABELS)
        with open(local_file, 'rb') as f:
            test_labels = extract_labels(f)
        return test_images, test_labels


if __name__ == "__main__":
    read_data_sets("/tmp/mnist/")
