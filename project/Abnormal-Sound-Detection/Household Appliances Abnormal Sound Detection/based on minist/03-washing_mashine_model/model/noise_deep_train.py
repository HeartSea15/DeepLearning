#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/12 2:04
# @Author  : Barry_J
# @Email   : s.barry1994@foxmail.com
# @File    : noise_deep_train.py
# @Software: PyCharm

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

# from swallowsound.swallowsound_input_data import read_data_sets
from swallowsound_input_data import read_data_sets
import tensorflow as tf
import time
import math

# 数据位置路径
# 数据50作为一段时，从时域到频域，只需要改三个地方：dir、deepnn中的x_image、x的tf.placeholder。
# 数据从50改为250时，改四个地方：第一个全连接两个地方的第二个维度改成2、dir、deepnn中的x_image、x的tf.placeholder

dir = './tmp/tensorflow/noise/input_2data250_fd/'

# 计算用时
def pass_time(time_):
    sum_time = math.floor(time_)

    h = math.floor(sum_time / 3600)
    m = math.floor((sum_time - h * 60 * 60) / 60)
    s = (sum_time - h * 60 * 60 - m * 60)

    print('\n用时时间:')
    print('hour:{0}  minute:{1}  second:{2}'.format(h, m, s))


# 定义变量和卷积函数
def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 定义池化层
def max_pool_1x5(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 1, 5, 1],
                          strides=[1, 1, 5, 1], padding='SAME')#步长5，池化的输出与输入保持同样的尺寸


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)#给权重制造一些随机的噪声来打破完全对称
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)#因为使用ReLU，也给偏置增加一些小的正值（0.1），用来避免死亡节点
    return tf.Variable(initial)


# 定义三个卷积层+可视化
############################################################
def deepnn(x):
    with tf.name_scope('resh_img'):
        x_image = tf.reshape(x, [-1, 1, 250, 1], name='Reshape')   # 时域50，频域25；

    # 第一层卷积层：
    with tf.name_scope('cov_layer1'):
        with tf.name_scope('conv1'):
            with tf.name_scope('W_conv1'):
                W_conv1 = weight_variable([1, 5, 1, 32])
                tf.summary.histogram('cov_layer1', W_conv1)
            with tf.name_scope('b_conv1'):
                b_conv1 = bias_variable([32])
                tf.summary.histogram('cov_layer1', b_conv1)
            with tf.name_scope('h_conv1'):
                h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # shape = (?, 1, 50, 32)
                tf.summary.histogram('cov_layer1', h_conv1)
            with tf.name_scope('B_CH1'):
                h_conv1 = tf.layers.batch_normalization(h_conv1, training=True)
        with tf.name_scope('pool_1'):
            h_pool1 = max_pool_1x5(h_conv1)  # shape = (?, 1, 10, 32)
            tf.summary.histogram('cov_layer1', h_pool1)

    # 第二层卷积层：
    with tf.name_scope('cov_layer2'):
        with tf.name_scope('conv2'):
            with tf.name_scope('W_conv2'):
                W_conv2 = weight_variable([1, 5, 32, 64])
                tf.summary.histogram('cov_layer2', W_conv2)
            with tf.name_scope('b_conv2'):
                b_conv2 = bias_variable([64])
                tf.summary.histogram('cov_layer2', b_conv2)
            with tf.name_scope('h_conv2'):
                h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # shape = (?, 1, 10, 64)
                tf.summary.histogram('cov_layer2', h_conv2)
            with tf.name_scope('B_CH2'):
                h_conv2 = tf.layers.batch_normalization(h_conv2, training=True)# 图片归一化，滑动平均，具体百度
        with tf.name_scope('pool_2'):
            h_pool2 = max_pool_1x5(h_conv2)  # shape = (?, 1, 2, 64)
            tf.summary.histogram('cov_layer2', h_pool2)

    # 第三层卷积层：
    with tf.name_scope('cov_layer3'):
        with tf.name_scope('conv3'):
            with tf.name_scope('W_conv3'):
                W_conv3 = weight_variable([1, 5, 64, 128])
                tf.summary.histogram('cov_layer3', W_conv3)
            with tf.name_scope('b_conv3'):
                b_conv3 = bias_variable([128])
                tf.summary.histogram('cov_layer3', b_conv3)
            with tf.name_scope('h_conv3'):
                h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)  # shape = (?, 1, 2, 128)
                tf.summary.histogram('cov_layer3', h_conv3)
            with tf.name_scope('B_CH3'):
                h_conv3 = tf.layers.batch_normalization(h_conv3,training=True)
        with tf.name_scope('pool_3'):
            h_pool3 = max_pool_1x5(h_conv3)  # shape = (?, 1, 1, 128)
            tf.summary.histogram('cov_layer3', h_pool3)

    ############################################################
    # 分别定义两个全连接层：
    # 第一层全连接层：
    with tf.name_scope('fc_layer1'):
        with tf.name_scope('fc1'):
            with tf.name_scope('W_fc1'):
                # 第三个卷积层的卷积核数量为128，其输出的tensor尺寸为1*2*128，1024个节点
                W_fc1 = weight_variable([1 * 2 * 128, 1024]) #改
                tf.summary.histogram('fc_layer1', W_fc1)
            with tf.name_scope('b_fc1'):
                b_fc1 = bias_variable([1024])
                tf.summary.histogram('fc_layer1', b_fc1)
            with tf.name_scope('h_pool3_flat'):
                h_pool3_flat = tf.reshape(h_pool3, [-1, 1 * 2 * 128])#变形，转成1D向量，改
                tf.summary.histogram('fc_layer1', h_pool3_flat)
            with tf.name_scope('h_fc1'):
                h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
                tf.summary.histogram('fc_layer1', h_fc1)
            with tf.name_scope('dropout'):
                keep_prob = tf.placeholder(tf.float32)
                h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)#使用一个dropout层
                tf.summary.histogram('fc_layer1', h_fc1_drop)

    # 第二层全连接层：
    with tf.name_scope('fc_layer2'):
        with tf.name_scope('fc2'):
            with tf.name_scope('W_fc2'):
                W_fc2 = weight_variable([1024, 2])
                tf.summary.histogram('fc_layer2', W_fc2)
            with tf.name_scope('b_fc2'):
                b_fc2 = bias_variable([2])
                tf.summary.histogram('fc_layer2', b_fc2)
            with tf.name_scope('y_conv'):
                y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
                tf.summary.histogram('fc_layer2', y_conv)
                return y_conv, keep_prob


# Import data
num_classes = 2
swallowsound = read_data_sets(dir,
                              gzip_compress=True,
                              train_imgaes='train-images-idx3-ubyte.gz',
                              train_labels='train-labels-idx1-ubyte.gz',
                              test_imgaes='t10k-images-idx3-ubyte.gz',
                              test_labels='t10k-labels-idx1-ubyte.gz',
                              one_hot=True,
                              validation_size=50,  ##意思：训练集(2万个)中用来验证的那部分（验证集，50个），剩下19950个用来训练
                              num_classes=num_classes,#分成2类
                              MSB=True)
print(swallowsound.train.images.shape, swallowsound.train.labels.shape)  # 训练集
print(swallowsound.test.images.shape, swallowsound.test.labels.shape)    # 测试集
print(swallowsound.validation.images.shape, swallowsound.validation.labels.shape)  # 验证集

# Create the model
with tf.name_scope('inputs'):
    x = tf.placeholder(tf.float32, [None, 250], name='x_input')  #时域50，频域25，改
    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 2], name='y_input')#真实的标签

# Build the graph for the deep net
y_conv, keep_prob = deepnn(x)

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    tf.summary.scalar('loss',cross_entropy)
with tf.name_scope('train_step'):
    train_step = tf.train.GradientDescentOptimizer(0.3).minimize(cross_entropy)  # accuracy 0.99
    # train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)   # accuracy 0.94
# AdamOptimizer(1e-4)
# GradientDescentOptimizer(0.5)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
correct_prediction = tf.cast(correct_prediction, tf.float32)
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(correct_prediction)
    tf.summary.scalar('accuracy',accuracy)

sess = tf.Session()
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('logs/', sess.graph)

sess.run(tf.global_variables_initializer())
print('训练开始！')
start = time.time()
for i in range(20000):
    batch = swallowsound.train.next_batch(500)
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    if i % 50 == 0:
        train_accuracy = accuracy.eval(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        print('输入第 %d 批batch, 此时的训练精度为：%g' % (i, train_accuracy))
        result = sess.run(merged, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        train_writer.add_summary(result, i)

pass_time(time.time() - start)
print('训练结束！')

print('#######################################################')
print('#######################################################')
print('#######################################################')

print('测试开始！')

print('测试精度为： %g' % accuracy.eval(session=sess, feed_dict={x: swallowsound.test.images, y_: swallowsound.test.labels, keep_prob: 0.5}))
print('测试结束！')

# 保存模型
saver=tf.train.Saver()
save_path=saver.save(sess, "3_cnn/model.ckpt")
print("训练模型保存成功，save to path",save_path)


#D:\pycharm\swallowsound\swallowsound\logs
#tensorboard  --logdir=D:\pycharm\swallowsound\swallowsound\logs\