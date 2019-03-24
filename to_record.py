# coding=utf-8
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import os
import math
import glob
import sys
from PIL import Image

dataset = './CamVid'
tfrecord_file = os.path.join(dataset, 'tfrecord')

_NUM_SHARDS = 2


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def image_to_example(raw_img_data, anno_img_data, filename, height, width):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(raw_img_data),
        'image/anno': bytes_feature(anno_img_data),
        'image/filename': bytes_feature(filename),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
    }))


def to_tfrecord(type):
    # write the image and annotation to the tfrecord file

    if not os.path.exists(tfrecord_file):
        os.mkdir(tfrecord_file)

    data_dir = os.path.join(dataset, type)
    img_filenames = glob.glob(os.path.join(data_dir, '*g'))
    anno_dir = os.path.join(dataset, type + 'annot')
    anno_filenames = glob.glob(os.path.join(anno_dir, '*g'))

    assert len(img_filenames) == len(anno_filenames)

    num_per_shard = int(math.ceil(len(anno_filenames) / _NUM_SHARDS))



    for shard_id in range(_NUM_SHARDS):
        output_tfrecord_filename = os.path.join(tfrecord_file, 'image_%s_%05d-of-%05d.tfrecord' % (type, shard_id, _NUM_SHARDS - 1))

        with tf.python_io.TFRecordWriter(output_tfrecord_filename) as tfrecord_writer:
            start_ndx = shard_id * num_per_shard
            end_ndx = min((shard_id + 1) * num_per_shard, len(anno_filenames))
            for i in range(start_ndx, end_ndx):
                try:
                    sys.stdout.write("\r>> Convert images %d/%d shard %d" % (i + 1, len(anno_filenames), shard_id))
                    sys.stdout.flush()

                    raw_img_data = Image.open(img_filenames[i])
                    raw_img_data_np = np.array(raw_img_data)
                    height, width, _ = raw_img_data_np.shape

                    anno_data = Image.open(anno_filenames[i])
                    anno_data_np = np.array(anno_data)
                    seg_height, seg_width = anno_data_np.shape

                    assert seg_height == height
                    assert seg_width == width

                    raw_img_data = raw_img_data.tobytes()
                    anno_data = anno_data.tobytes()
                    # print(anno_data)
                    example = image_to_example(raw_img_data, anno_data, os.path.basename(anno_filenames[i]), height, width)
                    tfrecord_writer.write(example.SerializeToString())
                except IOError as e:
                    print("Could not read: " + anno_filenames[i])
                    print("Error: " + e)
                    print("Skip it\n")
    print("\n %s data is ok" % type)


if __name__ == '__main__':

    to_tfrecord('train')
    to_tfrecord('test')
    to_tfrecord('val')
