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

BATCH_SIZE = 1
HEIGHT = 360
WIDTH = 480
saved_ckpt_path = './checkpoint/'
saved_prediction = './pred/'

classes = ['Sky', 'Building', 'Pole', 'Road', 'Pavement', 'Tree', 'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist','Background']
cmap = np.array([[128, 128, 128],
                [128, 0, 0],
                [192, 192, 192],
                [128, 64, 128],
                [60, 40, 222],
                [128, 128, 0],
                [192, 128, 128],
                [64, 64, 128],
                [64, 0, 128],
                [64, 64, 0],
                [0, 128, 192],
                [0, 0, 0]
                ])

def color_gray(image):
    height, width = image.shape

    return_img = np.zeros([height, width, 3], np.uint8)
    for i in range(height):
        for j in range(width):
            return_img[i, j, :] = cmap[image[i, j]]

    return return_img


image_batch, anno_batch, filename = input_data.read_batch(BATCH_SIZE, type='val')

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


        # save raw image, annotation, and prediction
        pred = pred.astype(np.uint8)
        b_anno = b_anno.astype(np.uint8)
        pred_color = color_gray(pred[0, :, :])
        b_anno_color = color_gray(b_anno[0, :, :])

        b_image *= 255
        b_image = b_image.astype(np.uint8)

        img = Image.fromarray(b_image[0])
        anno = Image.fromarray(b_anno_color)
        pred = Image.fromarray(pred_color)

        basename = b_filename[0].split('.')[0]
        #print(basename)

        if not os.path.exists(saved_prediction):
            os.mkdir(saved_prediction)
        img.save(os.path.join(saved_prediction, basename + '.png'))
        anno.save(os.path.join(saved_prediction, basename + '_anno.png'))
        pred.save(os.path.join(saved_prediction, basename + '_pred.png'))

        print("%s.png: prediction saved in %s" % (basename, saved_prediction))


        '''
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
        '''

    coord.request_stop()
    # 其他所有线程关闭后，这一函数才能返回
    coord.join(threads)