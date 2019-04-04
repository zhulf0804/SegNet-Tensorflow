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
saved_ckpt_path = './checkpoint/'
saved_summary_path = './summary/'


with tf.name_scope("input"):

    x = tf.placeholder(tf.float32, [BATCH_SIZE, HEIGHT, WIDTH, 3], name='x_input')
    y = tf.placeholder(tf.int32, [BATCH_SIZE, HEIGHT, WIDTH], name='ground_truth')

logits = Seg.segnet_2(x)

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits, name='loss'))
    tf.summary.scalar('loss', loss)

optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)

with tf.name_scope("mIoU"):
    softmax = tf.nn.softmax(logits, axis=-1)
    predictions = tf.argmax(logits, axis=-1, name='predictions')
    mIoU = tf.metrics.mean_iou(y, predictions, CLASSES, name='mIoU')
    tf.summary.scalar('mIoU', mIoU[0])


merged = tf.summary.merge_all()

image_batch, anno_batch, filename = input_data.read_batch(BATCH_SIZE, type = 'train')
image_batch_test, anno_batch_test, filename_test = input_data.read_batch(BATCH_SIZE, type = 'val')

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

    summary_writer = tf.summary.FileWriter(saved_summary_path, sess.graph)

    # input = np.random.rand(8, 256, 256, 3)
    # y_ = np.random.randint(CLASSES, size=8*256*256).reshape([8, 256, 256])


    for i in range(0, MAX_STEPS + 1):

        b_image, b_anno, b_filename = sess.run([image_batch, anno_batch, filename])
        b_image = b_image - 0.5
        b_image_test, b_anno_test, b_filename_test = sess.run([image_batch_test, anno_batch_test, filename_test])
        b_image_test = b_image_test - 0.5
        summary, _ = sess.run([merged, optimizer], feed_dict={x: b_image, y: b_anno})
        summary_writer.add_summary(summary, i)

        train_mIoU_val, train_loss_val = sess.run([mIoU, loss], feed_dict={x: b_image, y: b_anno })
        test_mIoU_val, test_loss_val = sess.run([mIoU, loss], feed_dict={x: b_image_test, y: b_anno_test})

        '''
        print(mIoU_val[0])
        print("**********************************")
        print(loss_val)
        print("**********************************")
        '''

        if i % 10 == 0:
            print("training step: %d, training loss: %f, training mIoU: %f, test loss: %f, test mIoU: %f" %(i, train_loss_val, train_mIoU_val[0], test_loss_val, test_mIoU_val[0]))

        if i % 2000 == 0:
            saver.save(sess, os.path.join(saved_ckpt_path, 'segnet.model'), global_step=i)


    coord.request_stop()
    # 其他所有线程关闭后，这一函数才能返回
    coord.join(threads)



