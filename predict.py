# coding=utf-8
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import segnet as Seg
import input_data

BATCH_SIZE = 4
HEIGHT = 360
WIDTH = 480
saved_ckpt_path = './checkpoint/'


image_batch, anno_batch, filename = input_data.read_batch(BATCH_SIZE)

with tf.name_scope("input"):

    x = tf.placeholder(tf.float32, [BATCH_SIZE, HEIGHT, WIDTH, 3], name='x_input')

logits = Seg.segnet(x)

with tf.name_scope('prediction'):
    prediction = tf.argmax(tf.nn.softmax(logits, axis=-1), axis=-1, name='prediction')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    #saver.restore(sess, './checkpoint/segnet.model-3000')
    ckpt = tf.train.get_checkpoint_state(saved_ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(1):
        b_image, b_anno, b_filename = sess.run([image_batch, anno_batch, filename])
        pred = sess.run(prediction, feed_dict={x: b_image})


        # visualization
        pred = pred.astype(np.uint8)
        b_image *= 255
        b_image = b_image.astype(np.uint8)
        for j in range(BATCH_SIZE):
            img = Image.fromarray(b_image[j])
            img_2 = (pred[j] * 10)
            img_2 = img_2.astype(np.uint8)
            anno = Image.fromarray(img_2)
            #anno_rgb = anno.convert("rgb")
            anno.save(str(j) + "_anno.png")
            img.save(str(j) + '_raw.png')
            plt.imshow(img)
            plt.axis('off')
            plt.show()
            plt.imshow(anno)
            plt.axis('off')
            plt.show()


    coord.request_stop()
    # 其他所有线程关闭后，这一函数才能返回
    coord.join(threads)