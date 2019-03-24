# coding=utf-8
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import os
import math
import sys
from PIL import Image

dataset = './CamVid'
tfrecord_file = os.path.join(dataset, 'tfrecord')

_NUM_SHARDS = 2
HEIGHT = 360
WIDTH = 480

def read_and_decode(filelist):
    filename_queue = tf.train.string_input_producer(filelist)
    reader = tf.TFRecordReader()
    _, serialized_exampe = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_exampe,
                                       features={
                                           'image/encoded': tf.FixedLenFeature([], tf.string),
                                           'image/anno': tf.FixedLenFeature([], tf.string),
                                           'image/filename': tf.FixedLenFeature([], tf.string),
                                           'image/height': tf.FixedLenFeature([], tf.int64),
                                           'image/width': tf.FixedLenFeature([], tf.int64),
                                       })

    image = tf.decode_raw(features['image/encoded'], tf.uint8)
    anno = tf.decode_raw(features['image/anno'], tf.uint8)
    filename =features['image/filename']
    height = tf.cast(features['image/height'], tf.int32)
    width = tf.cast(features['image/width'], tf.int32)

    image = tf.reshape(image, [HEIGHT, WIDTH, 3])
    anno = tf.reshape(anno, [HEIGHT, WIDTH])

    image = tf.cast(image, tf.float32)
    image = image / 255

    return image, anno, filename

def read_batch(batch_size, type = 'train'):
    filelist = [os.path.join(tfrecord_file, 'image_%s_%05d-of-%05d.tfrecord' % (type, shard_id, _NUM_SHARDS - 1)) for shard_id in range(_NUM_SHARDS)]
    print(filelist)
    image, anno, filename = read_and_decode(filelist)

    image_batch, anno_batch, filename = tf.train.shuffle_batch([image, anno, filename], batch_size=batch_size, capacity=128, min_after_dequeue=64, num_threads=2)

    # print(image_batch, anno_batch)

    return image_batch, anno_batch, filename

if __name__ == '__main__':
    BATCH_SIZE = 4
    image_batch, anno_batch, filename = read_batch(BATCH_SIZE, type='train')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        b_image, b_anno, b_filename = sess.run([image_batch, anno_batch, filename])
        print(b_image.shape)
        print(b_anno.shape)
        print(b_filename)

        b_image, b_anno, b_filename = sess.run([image_batch, anno_batch, filename])
        print(b_image.shape)
        print(b_anno.shape)
        print(b_filename)

        coord.request_stop()
        # 其他所有线程关闭后，这一函数才能返回
        coord.join(threads)