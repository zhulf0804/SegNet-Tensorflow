# coding=utf-8
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import math

CLASSES = 12
BATCH_SIZE = 4
batch_size = BATCH_SIZE
kernel_size = 7

def weight_variable(shape, stddev=None, name='weight'):
    if stddev == None:
        stddev = math.sqrt(2. / (shape[0] * shape[1] * shape[2]))
    else:
        stddev = 0.1
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)
def bias_variable(shape, name='bias'):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def bn_layer(x,is_training,name='BatchNorm',moving_decay=0.9,eps=1e-5):
    # 获取输入维度并判断是否匹配卷积层(4)或者全连接层(2)
    shape = x.shape
    assert len(shape) in [2,4]

    param_shape = shape[-1]
    with tf.variable_scope(name):
        # 声明BN中唯一需要学习的两个参数，y=gamma*x+beta
        gamma = tf.get_variable('gamma',param_shape,initializer=tf.constant_initializer(1))
        beta  = tf.get_variable('beat', param_shape,initializer=tf.constant_initializer(0))

        # 计算当前整个batch的均值与方差
        axes = list(range(len(shape)-1))
        batch_mean, batch_var = tf.nn.moments(x,axes,name='moments')

        # 采用滑动平均更新均值与方差
        ema = tf.train.ExponentialMovingAverage(moving_decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean,batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        # 训练时，更新均值与方差，测试时使用之前最后一次保存的均值与方差
        mean, var = tf.cond(tf.equal(is_training,True),mean_var_with_update,
                lambda:(ema.average(batch_mean),ema.average(batch_var)))

        # 最后执行batch normalization
        return tf.nn.batch_normalization(x,mean,var,beta,gamma,eps)


def conv_bn_relu(input, w, b, i, name='conv_bn_relu', bias=False, bn=True, relu=True):

    conv_2d = tf.nn.conv2d(input, w, [1, 1, 1, 1], padding='SAME', name='conv')

    if bias:
        conv_2d = tf.nn.bias_add(conv_2d, b, name='bias')
    if bn:
        conv_2d = bn_layer(conv_2d, True, name='BN_%d' %i)

    if relu:
        conv_2d = tf.nn.relu(conv_2d, name='relu')

    return conv_2d


def unpool_with_argmax(bottom, argmax, output_shape=None, name='max_unpool_with_argmax'):
    '''
    upsampling according argmax
    :param bottom: the output feature maps needed to be upsampled
    :param argmax: the indice made by tf.nn.max_pool_with_argmax()
    :param output_shape:
    :param name:
    :return:
    '''
    with tf.name_scope(name):
        ksize = [1, 2, 2, 1]
        input_shape = bottom.get_shape().as_list()
        #print(input_shape)
        #  calculation new shape
        if output_shape is None:
            output_shape = (input_shape[0],
                            input_shape[1] * ksize[1],
                            input_shape[2] * ksize[2],
                            input_shape[3])
        flat_input_size = np.prod(input_shape)
        flat_output_size = np.prod(output_shape)
        bottom_ = tf.reshape(bottom, [flat_input_size])
        argmax_ = tf.reshape(argmax, [flat_input_size, 1])

        ret = tf.scatter_nd(argmax_, bottom_, [flat_output_size])

        ret = tf.reshape(ret, output_shape)
        return ret

def un_conv(input, num_input_channels, conv_filter_size, num_filters, height, width, train=True, padding='SAME',relu=True):


    weights = weight_variable(shape=[conv_filter_size, conv_filter_size, num_filters, num_input_channels])
    biases = bias_variable([num_filters])
    if train:
        batch_size_0 = batch_size
    else:
        batch_size_0 = 1
    layer = tf.nn.conv2d_transpose(value=input, filter=weights,
                                   output_shape=[batch_size_0, height, width, num_filters],
                                   strides=[1, 2, 2, 1],
                                   padding=padding)
    layer += biases

    if relu:
        layer = tf.nn.relu(layer)
    return layer

base_feature_num = 64


def segnet(input):

    with tf.name_scope("encode"):

        with tf.name_scope("layer_1"):
            cur_feature_num = base_feature_num
            w = weight_variable(shape=[kernel_size, kernel_size, 3, cur_feature_num])
            b = bias_variable(shape=[cur_feature_num])
            conv = conv_bn_relu(input, w, b, 1) # i is used for variable_scope

        input = conv
        with tf.name_scope("layer_2"):
            cur_feature_num = base_feature_num
            w = weight_variable(shape=[kernel_size, kernel_size, cur_feature_num, cur_feature_num])
            b = bias_variable(shape=[cur_feature_num])
            conv = conv_bn_relu(input, w, b, 2) # i is used for variable_scope
            pool, pool1_indices = tf.nn.max_pool_with_argmax(conv, [1, 2, 2, 1], padding='SAME', strides=[1, 2, 2, 1], name='pool')

        input = pool

        for i in range(3, 5):
            with tf.name_scope("layer_%d" %i):
                cur_feature_num = base_feature_num * 2
                if i == 3:
                    w = weight_variable(shape=[kernel_size, kernel_size, cur_feature_num // 2, cur_feature_num])
                else:
                    w = weight_variable(shape=[kernel_size, kernel_size, cur_feature_num, cur_feature_num])
                b = bias_variable(shape=[cur_feature_num])
                conv = conv_bn_relu(input, w, b, i)  # i is used for variable_scope
                if i == 4:
                    pool, pool2_indices = tf.nn.max_pool_with_argmax(conv, [1, 2, 2, 1], padding='SAME', strides=[1, 2, 2, 1], name='pool')
                    input = pool
                else:
                    input = conv

        for i in range(5, 8):
            with tf.name_scope("layer_%d" %i):
                cur_feature_num = base_feature_num * 4
                if i == 5:
                    w = weight_variable(shape=[kernel_size, kernel_size, cur_feature_num // 2, cur_feature_num])
                else:
                    w = weight_variable(shape=[kernel_size, kernel_size, cur_feature_num, cur_feature_num])
                b = bias_variable(shape=[cur_feature_num])
                conv = conv_bn_relu(input, w, b, i)  # i is used for variable_scope
                if i == 7:
                    pool, pool3_indices = tf.nn.max_pool_with_argmax(conv, [1, 2, 2, 1], padding='SAME', strides=[1, 2, 2, 1], name='pool')
                    input = pool
                else:
                    input = conv

        for i in range(8, 11):
            with tf.name_scope("layer_%d" %i):
                cur_feature_num = base_feature_num * 8
                if i == 8:
                    w = weight_variable(shape=[kernel_size, kernel_size, cur_feature_num // 2, cur_feature_num])
                else:
                    w = weight_variable(shape=[kernel_size, kernel_size, cur_feature_num, cur_feature_num])
                b = bias_variable(shape=[cur_feature_num])
                conv = conv_bn_relu(input, w, b, i)  # i is used for variable_scope
                if i == 10:
                    pool, pool4_indices = tf.nn.max_pool_with_argmax(conv, [1, 2, 2, 1], padding='SAME', strides=[1, 2, 2, 1], name='pool')
                    input = pool
                else:
                    input = conv

        for i in range(11, 14):
            with tf.name_scope("layer_%d" %i):
                cur_feature_num = base_feature_num * 8
                w = weight_variable(shape=[kernel_size, kernel_size, cur_feature_num, cur_feature_num])
                b = bias_variable(shape=[cur_feature_num])
                conv = conv_bn_relu(input, w, b, i)  # i is used for variable_scope
                input = conv


    with tf.name_scope("decode"):
        for i in range(1, 4):
            with tf.name_scope("layer_%d" % i):
                cur_feature_num = base_feature_num * 8

                w = weight_variable(shape=[kernel_size, kernel_size, cur_feature_num, cur_feature_num])
                b = bias_variable(shape=[cur_feature_num])
                conv = conv_bn_relu(input, w, b, i + 13, relu=False)  # i is used for variable_scope

                if i == 3:
                    up_sample = unpool_with_argmax(conv, pool4_indices, name='unpool1')
                    ## for CamVid dataset
                    input_shape = up_sample.get_shape().as_list()
                    input = tf.slice(up_sample, [0, 0, 0, 0], [input_shape[0], input_shape[1] - 1, input_shape[2], input_shape[3]])
                else:
                    input = conv

                # print(input)
        for i in range(4, 7):
            with tf.name_scope("layer_%d" % i):
                cur_feature_num = base_feature_num * 8
                if i == 6:
                    w = weight_variable(shape=[kernel_size, kernel_size, cur_feature_num, cur_feature_num//2])
                    b = bias_variable(shape=[cur_feature_num//2])
                else:
                    w = weight_variable(shape=[kernel_size, kernel_size, cur_feature_num, cur_feature_num])
                    b = bias_variable(shape=[cur_feature_num])
                conv = conv_bn_relu(input, w, b, i + 13, relu=False)  # i is used for variable_scope

                if i == 6:
                    up_sample = unpool_with_argmax(conv, pool3_indices, name='unpool2')
                    input = up_sample
                else:
                    input = conv
        for i in range(7, 10):
            with tf.name_scope("layer_%d" % i):
                cur_feature_num = base_feature_num * 4
                if i == 9:
                    w = weight_variable(shape=[kernel_size, kernel_size, cur_feature_num, cur_feature_num//2])
                    b = bias_variable(shape=[cur_feature_num//2])
                else:
                    w = weight_variable(shape=[kernel_size, kernel_size, cur_feature_num, cur_feature_num])
                    b = bias_variable(shape=[cur_feature_num])
                conv = conv_bn_relu(input, w, b, i + 13, relu=False)  # i is used for variable_scope

                if i == 9:
                    up_sample = unpool_with_argmax(conv, pool2_indices, name='unpool3')
                    input = up_sample
                else:
                    input = conv
        for i in range(10, 12):
            with tf.name_scope("layer_%d" % i):
                cur_feature_num = base_feature_num * 2
                if i == 11:
                    w = weight_variable(shape=[kernel_size, kernel_size, cur_feature_num, cur_feature_num//2])
                    b = bias_variable(shape=[cur_feature_num//2])
                else:
                    w = weight_variable(shape=[kernel_size, kernel_size, cur_feature_num, cur_feature_num])
                    b = bias_variable(shape=[cur_feature_num])
                conv = conv_bn_relu(input, w, b, i + 13, relu=False)  # i is used for variable_scope

                if i ==11:
                    up_sample = unpool_with_argmax(conv, pool1_indices, name='unpool5')
                    input = up_sample
                else:
                    input = conv
        for i in range(12, 14):
            with tf.name_scope("layer_%d" % i):
                cur_feature_num = base_feature_num

                w = weight_variable(shape=[kernel_size, kernel_size, cur_feature_num, cur_feature_num])
                b = bias_variable(shape=[cur_feature_num])
                conv = conv_bn_relu(input, w, b, i + 13, relu=False)  # i is used for variable_scope

                input = conv

    with tf.name_scope('output'):

        w = weight_variable(shape=[kernel_size, kernel_size, cur_feature_num, CLASSES])
        b = bias_variable(shape=[CLASSES])
        conv = conv_bn_relu(input, w, b, i=222, bn=False, relu=False)
        input = conv

    return input

def segnet_2(input, train=True):
    with tf.name_scope("encode"):

        with tf.name_scope("layer_1"):
            cur_feature_num = base_feature_num
            w = weight_variable(shape=[kernel_size, kernel_size, 3, cur_feature_num])
            b = bias_variable(shape=[cur_feature_num])
            conv = conv_bn_relu(input, w, b, 1) # i is used for variable_scope

        input = conv
        with tf.name_scope("layer_2"):
            cur_feature_num = base_feature_num
            w = weight_variable(shape=[kernel_size, kernel_size, cur_feature_num, cur_feature_num])
            b = bias_variable(shape=[cur_feature_num])
            conv = conv_bn_relu(input, w, b, 2) # i is used for variable_scope
            pool, pool1_indices = tf.nn.max_pool_with_argmax(conv, [1, 2, 2, 1], padding='SAME', strides=[1, 2, 2, 1], name='pool')

        input = pool

        for i in range(3, 5):
            with tf.name_scope("layer_%d" %i):
                cur_feature_num = base_feature_num * 2
                if i == 3:
                    w = weight_variable(shape=[kernel_size, kernel_size, cur_feature_num // 2, cur_feature_num])
                else:
                    w = weight_variable(shape=[kernel_size, kernel_size, cur_feature_num, cur_feature_num])
                b = bias_variable(shape=[cur_feature_num])
                conv = conv_bn_relu(input, w, b, i)  # i is used for variable_scope
                if i == 4:
                    pool, pool2_indices = tf.nn.max_pool_with_argmax(conv, [1, 2, 2, 1], padding='SAME', strides=[1, 2, 2, 1], name='pool')
                    input = pool
                else:
                    input = conv

        for i in range(5, 8):
            with tf.name_scope("layer_%d" %i):
                cur_feature_num = base_feature_num * 4
                if i == 5:
                    w = weight_variable(shape=[kernel_size, kernel_size, cur_feature_num // 2, cur_feature_num])
                else:
                    w = weight_variable(shape=[kernel_size, kernel_size, cur_feature_num, cur_feature_num])
                b = bias_variable(shape=[cur_feature_num])
                conv = conv_bn_relu(input, w, b, i)  # i is used for variable_scope
                if i == 7:
                    pool, pool3_indices = tf.nn.max_pool_with_argmax(conv, [1, 2, 2, 1], padding='SAME', strides=[1, 2, 2, 1], name='pool')
                    input = pool
                else:
                    input = conv

        for i in range(8, 11):
            with tf.name_scope("layer_%d" %i):
                cur_feature_num = base_feature_num * 8
                if i == 8:
                    w = weight_variable(shape=[kernel_size, kernel_size, cur_feature_num // 2, cur_feature_num])
                else:
                    w = weight_variable(shape=[kernel_size, kernel_size, cur_feature_num, cur_feature_num])
                b = bias_variable(shape=[cur_feature_num])
                conv = conv_bn_relu(input, w, b, i)  # i is used for variable_scope
                if i == 10:
                    pool, pool4_indices = tf.nn.max_pool_with_argmax(conv, [1, 2, 2, 1], padding='SAME', strides=[1, 2, 2, 1], name='pool')
                    input = pool
                else:
                    input = conv

        for i in range(11, 14):
            with tf.name_scope("layer_%d" %i):
                cur_feature_num = base_feature_num * 8
                w = weight_variable(shape=[kernel_size, kernel_size, cur_feature_num, cur_feature_num])
                b = bias_variable(shape=[cur_feature_num])
                conv = conv_bn_relu(input, w, b, i)  # i is used for variable_scope
                input = conv


    with tf.name_scope("decode"):
        for i in range(1, 4):
            with tf.name_scope("layer_%d" % i):
                cur_feature_num = base_feature_num * 8

                w = weight_variable(shape=[kernel_size, kernel_size, cur_feature_num, cur_feature_num])
                b = bias_variable(shape=[cur_feature_num])
                conv = conv_bn_relu(input, w, b, i + 13, relu=False)  # i is used for variable_scope
                if i == 3:
                    shape = conv.get_shape().as_list()
                    up_sample = un_conv(conv, cur_feature_num ,kernel_size, cur_feature_num, 2*shape[1] - 1, 2 * shape[2], train)
                    input = up_sample
                else:
                    input = conv

                # print(input)
        for i in range(4, 7):
            with tf.name_scope("layer_%d" % i):
                cur_feature_num = base_feature_num * 8
                if i == 6:
                    shape = conv.get_shape().as_list()
                    up_sample = un_conv(conv, cur_feature_num, kernel_size, cur_feature_num // 2, 2 * shape[1], 2 * shape[2], train)
                    input = up_sample
                else:
                    w = weight_variable(shape=[kernel_size, kernel_size, cur_feature_num, cur_feature_num])
                    b = bias_variable(shape=[cur_feature_num])
                    conv = conv_bn_relu(input, w, b, i + 13, relu=False)  # i is used for variable_scope
                    input =conv

        for i in range(7, 10):
            with tf.name_scope("layer_%d" % i):
                cur_feature_num = base_feature_num * 4
                if i == 9:
                    shape = conv.get_shape().as_list()
                    up_sample = un_conv(conv, cur_feature_num, kernel_size, cur_feature_num // 2, 2 * shape[1],
                                        2 * shape[2], train)
                    input = up_sample
                else:
                    w = weight_variable(shape=[kernel_size, kernel_size, cur_feature_num, cur_feature_num])
                    b = bias_variable(shape=[cur_feature_num])
                    conv = conv_bn_relu(input, w, b, i + 13, relu=False)  # i is used for variable_scope
                    input = conv

        for i in range(10, 12):
            with tf.name_scope("layer_%d" % i):
                cur_feature_num = base_feature_num * 2
                if i == 11:
                    shape = conv.get_shape().as_list()
                    up_sample = un_conv(conv, cur_feature_num, kernel_size, cur_feature_num // 2, 2 * shape[1],
                                        2 * shape[2], train)
                    input = up_sample
                else:
                    w = weight_variable(shape=[kernel_size, kernel_size, cur_feature_num, cur_feature_num])
                    b = bias_variable(shape=[cur_feature_num])
                    conv = conv_bn_relu(input, w, b, i + 13, relu=False)  # i is used for variable_scope
                    input = conv

        for i in range(12, 14):
            with tf.name_scope("layer_%d" % i):
                cur_feature_num = base_feature_num

                w = weight_variable(shape=[kernel_size, kernel_size, cur_feature_num, cur_feature_num])
                b = bias_variable(shape=[cur_feature_num])
                conv = conv_bn_relu(input, w, b, i + 13, relu=False)  # i is used for variable_scope

                input = conv

    with tf.name_scope('output'):

        w = weight_variable(shape=[kernel_size, kernel_size, cur_feature_num, CLASSES])
        b = bias_variable(shape=[CLASSES])
        conv = conv_bn_relu(input, w, b, i=222, bn=False, relu=False)
        input = conv

    return input

if __name__ == '__main__':
    input = tf.constant(1.0, shape=[BATCH_SIZE, 360, 480, 3])
    with tf.Session() as sess:
        y = segnet_2(input)

        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        for i in range(2):

            print(y)
            y_val = sess.run(y)
            print(y_val)


