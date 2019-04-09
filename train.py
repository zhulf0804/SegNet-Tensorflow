# coding=utf-8
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import os
import segnet as Seg
import input_data

CLASSES = Seg.CLASSES
MAX_STEPS = 20000
HEIGHT = input_data.HEIGHT
WIDTH = input_data.WIDTH
BATCH_SIZE = 4
scale = Seg.scale
saved_ckpt_path = './checkpoint/'
saved_summary_train_path = './summary/train/'
saved_summary_test_path = './summary/test/'


with tf.name_scope("input"):

    x = tf.placeholder(tf.float32, [BATCH_SIZE, HEIGHT, WIDTH, 3], name='x_input')
    y = tf.placeholder(tf.int32, [BATCH_SIZE, HEIGHT, WIDTH], name='ground_truth')

logits = Seg.segnet_2(x)

with tf.name_scope('regularization'):
    regularizer = tf.contrib.layers.l2_regularizer(scale)
    reg_term = tf.contrib.layers.apply_regularization(regularizer)

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits, name='loss'))
    loss_all = loss + reg_term
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('loss_all', loss_all)

optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss_all)

with tf.name_scope("mIoU"):
    softmax = tf.nn.softmax(logits, axis=-1)
    predictions = tf.argmax(logits, axis=-1, name='predictions')
    mIoU_train = tf.metrics.mean_iou(y, predictions, CLASSES, name='mIoU_train')
    tf.summary.scalar('mIoU_train', mIoU_train[0])
    mIoU_test = tf.metrics.mean_iou(y, predictions, CLASSES, name='mIoU_test')
    tf.summary.scalar('mIoU_test', mIoU_train[0])


merged = tf.summary.merge_all()

image_batch, anno_batch, filename = input_data.read_batch(BATCH_SIZE, type = 'trainval')
image_batch_test, anno_batch_test, filename_test = input_data.read_batch(BATCH_SIZE, type = 'test')

with tf.Session() as sess:

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    #if os.path.exists(saved_ckpt_path):
    ckpt = tf.train.get_checkpoint_state(saved_ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    #saver.restore(sess, './checkpoint/segnet.model-30000')

    train_summary_writer = tf.summary.FileWriter(saved_summary_train_path, sess.graph)
    test_summary_writer = tf.summary.FileWriter(saved_summary_test_path, sess.graph)

    # input = np.random.rand(8, 256, 256, 3)
    # y_ = np.random.randint(CLASSES, size=8*256*256).reshape([8, 256, 256])


    for i in range(0, MAX_STEPS + 1):

        b_image, b_anno, b_filename = sess.run([image_batch, anno_batch, filename])
        b_image = b_image - 0.5
        b_image_test, b_anno_test, b_filename_test = sess.run([image_batch_test, anno_batch_test, filename_test])
        b_image_test = b_image_test - 0.5

        train_summary, _ = sess.run([merged, optimizer], feed_dict={x: b_image, y: b_anno})
        train_summary_writer.add_summary(train_summary, i)
        test_summary = sess.run(merged, feed_dict={x: b_image_test, y: b_anno_test})
        test_summary_writer.add_summary(test_summary, i)

        train_mIoU_val, train_loss_val_all, train_loss_val = sess.run([mIoU_train, loss_all, loss], feed_dict={x: b_image, y: b_anno })
        test_mIoU_val, test_loss_val_all, test_loss_val = sess.run([mIoU_test, loss_all, loss], feed_dict={x: b_image_test, y: b_anno_test})

        if i % 10 == 0:
            print("training step: %d, training loss all: %f, training loss: %f, training mIoU: %f, test loss all: %f, test loss: %f, test mIoU: %f" %(i, train_loss_val_all, train_loss_val, train_mIoU_val[0], test_loss_val_all, test_loss_val, test_mIoU_val[0]))

        if i % 2000 == 0:
            saver.save(sess, os.path.join(saved_ckpt_path, 'segnet.model'), global_step=i)


    coord.request_stop()
    coord.join(threads)
