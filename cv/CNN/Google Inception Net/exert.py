"""
    作者：Heart Sea
    结构：由于Google Inception Net V3相对复杂，使用tf.contrib.slim辅助设计这个网络。
    contrib.slim中的一些功能和组件可以大大减少设计Inception Net的代码量，
    我们只需要少量代码即可构建好有42层深的Inception V3。

    model：
        类型     kernel尺寸 / 步长     输出尺寸
  input 299x299x3
        卷积       3x3 / 2           149x149x32
        卷积       3x3 / 1           147x147x32
        卷积       3x3 / 1           147x147x64
        池化       3x3 / 2           73x73x64
        卷积       1x1 / 1           73x73x80
        卷积       3x3 / 1           71x71x192
        池化       3x3 / 2           35x35x192

    inception 模块组      3个Inception Module  35x35x288
    inception 模块组      5个Inception Module  17x17x768
    inception 模块组      3个Inception Module  8x8x2048

        池化          8x8            1x1x2048
        dropout      0.8             1x1x2048
        卷积         1x1             1x1x1000
        线性       logits            1000
        softmax    分类输出          1000

    数据集：不使用ImageNet数据集来训练， 构造出Google Inception Net V3网络结构，并评测前馈计算耗时
    日期：12/07/2019
    总结：gpu，forword平均每轮耗时0.043，比VGGNet的0.07s更快，这主要归功于较小的参数量。
"""

import tensorflow as tf
from datetime import datetime
import math
import time
slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)  # 产生截断的正太分布


def inception_v3_arg_scope(weight_decay=0.00004,
                           stddev=0.1,
                           batch_norm_var_collection='moving_vars'):
    """
    生成网络中经常用到的函数的默认参数，如卷积的激活函数、权重初始化方式、标准化器等。
    :param weight_decay:    L2正则的weight_decay为0.00004
    :param stddev:          标准差为0.1
    :param batch_norm_var_collection:   moving_vars
    :return:
    """
    # 定义批量归一化（batch_normalization）的参考字典
    batch_norm_params = {
        'decay': 0.9997,                                   # 衰减系数
        'epsilon': 0.001,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'variables_collections': {
            'beta': None,
            'gamma': None,
            'moving_mean': [batch_norm_var_collection],
            'moving_variance': [batch_norm_var_collection],
        }
    }
    # 使用slim.arg_scope,这是个非常有用的工具，他可以给函数的参数自动赋予某些默认值
    # 使用slim.arg_scope后就不要每次重复设置参数。只需要在修改的地方设置。
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope([slim.conv2d],
                            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),  # 权重初始化
                            activation_fn=tf.nn.relu,       # 激活函数
                            normalizer_fn=slim.batch_norm,  # 标准化
                            normalizer_params=batch_norm_params) as sc:  # 标准化器的参数
            return sc


def inception_v3_base(inputs, scope=None):
    """
    生成inception V3网络的卷积部分，三个连续的inception模块组部分
    :param inputs: 输入图片数据的tensor
    :param scope: 包含了函数默认参数的环境
    :return: 最后一个inception模块组的最后一个Inception Module，和第二个Inception模块组的最后一个Inception Module
    总结：从开始的299x299x3到8x8x2048，每一层卷积、池化或inception模块组的目的是将空间结构简化，同时将空间信息转化为高阶抽象的特征信息，
        即把空间的维度转化为通道的维度，这一过程同时也使每层输出tensor的总size持续下降，降低计算量。
    """
    end_points = {}  # 保存某些关键节点供之后使用
    with tf.variable_scope(scope, 'InceptionV3', [inputs]):
        # 将3个函数的参数设置默认值
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1,
                            padding='VALID'):
            # 开始定义InceptionV3
            # inputs 299x299x3
            net = slim.conv2d(inputs, 32, [3, 3], stride=2, scope='Conv2d_1a_3x3')     # 149x149x32
            net = slim.conv2d(net, 32, [3, 3], scope='Conv2d_2a_3x3')                  # 147x147x32
            net = slim.conv2d(net, 64, [3, 3], padding='SAME', scope='Conv2d_2b_3x3')  # 147x147x64
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='Maxpool_3a_3x3')       # 73x73x64
            net = slim.conv2d(net, 80, [1, 1], scope='Conv2d_3b_1x1')                  # 73x73x80
            net = slim.conv2d(net, 192, [3, 3], scope='Conv2d_4a_3x3')                 # 71x71x192
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='maxpool_5a_3x3')       # 35x35x192
            # 总结，以上几个非inception module的卷积层中，主要使用了3x3的小卷积核，充分借鉴了VGGNet的结构
            # 实现了对图片数据的压缩，并对图片特征进行了抽象。
            # 1x1卷积，低成本的跨通道的对特征进行组合

        # 接下来是三个连续的inception模块组，各自有多个inception module,
        # 每个模块组内部的inception module结构非常类似，有些细节不同
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='SAME'):
            """
                定义第一个inception模块组, 3个inception module
            """
            # 定义第一个inception模块组的第一个inception module, 有4个分支
            with tf.variable_scope('Mixed_5b'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0b_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)  # 第三个维度合并，即输出通道合并
                # 总结：通道数=64+64+96+32=256， 图片尺寸35x35， tensor尺寸=35x35x256

            # 定义第一个inception模块组的第2个inception module，有4个分支
            # 与第一个inception module不同的是第四个分支的最后接的是64通道的1x1的卷积
            with tf.variable_scope('Mixed_5c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0b_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0c_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)  # 第三个维度合并，即输出通道合并
                # 总结：通道数=64+64+96+64=288， 图片尺寸35x35， tensor尺寸=35x35x288

            # 定义第一个inception模块组的第3个inception module，有4个分支
            # 与第2个inception module完全相同
            with tf.variable_scope('Mixed_5d'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0b_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)  # 第三个维度合并，即输出通道合并
                # 总结：通道数=64+64+96+64=288， 图片尺寸35x35， tensor尺寸=35x35x288

            """
                定义第二个inception模块组,5个inception module
            """
            # 第二个inception模块组的第1个inception module，有3个分支
            with tf.variable_scope('Mixed_6a'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 384, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_1x1')

                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_1 = slim.conv2d(branch_1, 96, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_1x1')

                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='MaxPool_1a_3x3')
                net = tf.concat([branch_0, branch_1, branch_2], 3)  # 第三个维度合并，即输出通道合并
                # 总结：tensor尺寸=17x17x(384+96+288)=17x17x768

            # 第二个Inception模块组的第2个Inception Module，有4个分支
            with tf.variable_scope('Mixed_6b'):
                # 第一个分支
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                # 第二个分支
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 128, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                # 第三个分支
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 128, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                # 第四个分支
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                # 总结：将四个分支合并，tensor的尺寸为17*17*(192+192+192+192)=17*17*768

            # 第二个Inception模块组的第3个Inception Module，有4个分支
            with tf.variable_scope('Mixed_6c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                # 总结：tensor的尺寸为17*17*(192+192+192+192)=17*17*768

            # 第二个Inception模块组的第4个Inception Module，有4个分支
            # 同第3个Inception Module
            with tf.variable_scope('Mixed_6d'):
                # 第一个分支
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                # 第二个分支
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                # 第三个分支
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                # 第四个分支
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                # 总结：将四个分支合并，tensor的尺寸为17*17*(192+192+192+192)=17*17*768

            # 第二个Inception模块组的第5个Inception Module，有4个分支
            # 同第3个Inception Module
            with tf.variable_scope('Mixed_6e'):
                # 第一个分支
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                # 第二个分支
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                # 第三个分支
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                # 第四个分支
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                # 总结：将四个分支合并，tensor的尺寸为17*17*(192+192+192+192)=17*17*768
            # 将Mixed_6e存储于end_points中，作为Auxiliary Classifier辅助模型的分类
            end_points['Mixed_6e'] = net

            """
                定义第三个inception模块组, 3个inception module
            """

            # 第三个Inception模块组的第一个Inception Module，有三个分支
            with tf.variable_scope('Mixed_7a'):
                # 第一个分支
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_0 = slim.conv2d(branch_0, 320, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                # 第二个分支
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                    branch_1 = slim.conv2d(branch_1, 192, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                # 第三个分支
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='MaxPool_1a_3x3')
                # 将三个分支合并，步长为2，图片尺寸变为原来的一半，所以tensor的尺寸为8*8*(320+192+768)=8*8*1280
                net = tf.concat([branch_0, branch_1, branch_2], 3)

            # 第三个Inception模块组的第二个Inception Module，有四个分支
            with tf.variable_scope('Mixed_7b'):
                # 第一个分支
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
                # 第二个分支
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = tf.concat([
                        slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'),
                        slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0b_3x1')], 3)  # 8*8*(384+384)=8*8*768
                # 第三个分支
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = tf.concat([
                        slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'),
                        slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')], 3)  # 8*8*(384+384)=8*8*768
                # 第四个分支
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                # 将四个分支合并,则tensor的尺寸为8*8*(320+768+768+192)=8*8*2048
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            # 第三个Inception模块组的第三个Inception Module，有四个分支
            # 同第二个Inception Module
            with tf.variable_scope('Mixed_7c'):
                # 第一个分支
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
                # 第二个分支
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = tf.concat([
                        slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'),
                        slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0b_3x1')], 3)  # 8*8*(384+384)=8*8*768
                # 第三个分支
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = tf.concat([
                        slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'),
                        slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')], 3)  # 8*8*(384+384)=8*8*768
                # 第四个分支
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                # 将四个分支合并,则tensor的尺寸为8*8*(320+768+768+192)=8*8*2048
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            # 返回这个Inception Module的结果作为该函数的结果
            return net, end_points


def inception_v3(inputs,
                 num_classes=1000,
                 is_training=True,
                 dropout_keep_prob=0.8,
                 prediction_fn=slim.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='InceptionV3'):
    """
    实现Inceptio V3网络的最后一部分---全局平均池化，Softmax 和 Auxiliary Logits
    :param inputs:
    :param num_classes: 最后需要分类的数目
    :param is_training: 是否是训练过程，只有训练时，Batch Normalization和Dropout才会被启用
    :param dropout_keep_prob: 训练时Dropout所需保留节点的比例，默认为0.8
    :param prediction_fn: 最后用来分类的函数，默认softmax
    :param spatial_squeeze: 是否对输出进行squeeze操作(即去除维数为1的维度，如5*5*1转为5*5)
    :param reuse: 是否会对网络和variable进行重复使用
    :param scope: 包含了了函数默认参数的环境
    :return:
    """
    # 定义网络的name和reuse等参数的默认值
    with tf.variable_scope(scope, 'InceptionV3', [inputs, num_classes], reuse=reuse) as scope:
        # 定义Batch Normalization和Dropout的is_training标志的默认值
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            # 使用该定义好的函数得到整个网络的卷积部分，得到返回
            net, end_points = inception_v3_base(inputs, scope=scope)

            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):

                '''处理辅助分类的节点Auxiliary Logits'''
                aux_logits = end_points['Mixed_6e']  # 取到Mixed_6e，tensor形状为17*17*768
                with tf.variable_scope('AuxLogits'):
                    aux_logits = slim.avg_pool2d(aux_logits, [5, 5], stride=3, padding='VALID',
                                                 scope='Conv2d_1b_1x1')  # 5*5*768

                    aux_logits = slim.conv2d(aux_logits, 128, [1, 1], scope='Conv2d_1x1')  # 5*5*128

                    aux_logits = slim.conv2d(aux_logits, 768, [5, 5], weights_initializer=trunc_normal(0.01),
                                             padding='VALID',
                                             scope='Conv2d_2a_5x5')  # 1*1*768

                    aux_logits = slim.conv2d(aux_logits, num_classes, [1, 1],
                                             activation_fn=None,
                                             normalizer_fn=None,
                                             weights_initializer=trunc_normal(0.001),
                                             scope='Conv2d_2b_1x1')  # 输出1*1*1000

                    if spatial_squeeze:  # 将tensor 1*1*1000中前两个为1的维度消除
                        aux_logits = tf.squeeze(aux_logits, [1, 2], name='SpatialSqueeze')
                    end_points['AuxLogits'] = aux_logits

                '''处理正常的分类预测逻辑'''
                with tf.variable_scope('Logits'):  # 8*8*2048
                    # 全局平均池化，这里相当于fc，
                    net = slim.avg_pool2d(net, [8, 8], padding='VALID', scope='AvgPool_1a_8x8')  # 输出1*1*2048
                    # Dropout层，虽然移除了全连接，但是网络中依然使用了Dropout
                    net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')

                    end_points['PreLogits'] = net
                    # 输出1*1*1000
                    logits = slim.conv2d(net, num_classes, [1, 1],
                                         activation_fn=None,
                                         normalizer_fn=None,
                                         scope='Conv2d_1c_1x1')

                    # 线性化，将tensor 1*1*1000中前两个为1的维度消除
                    if spatial_squeeze:
                        logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')

                end_points['Logits'] = logits
                # Softmax分类器对结果进行分类预测
                end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    # 返回输出结果和包含辅助节点的end_points
    return logits, end_points


def time_tensorflow_run(session, target, info_string):
    """
    :param session: tensorflow 的session
    :param target: 需要评测的运算算子，这里是卷积网络最后一个池化层的输出pool5
    :param info_string: 测试的名称
    :return: 评估每轮迭代耗时
    """
    num_steps_burn_in = 10  # 预热轮数，只考两10轮以后的计算时间,头几轮迭代有显存的加载、cache命中等问题因此可以跳过
    total_duration = 0.0    # 总时间
    total_duration_squared = 0.0  # 总时间平方和
    for i in range(num_steps_burn_in+num_batches):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time()-start_time
        if i >= num_steps_burn_in:
            if not i % 10:  # 每10轮迭代显示当前迭代所需要的时间
                print('%s:step %d,duration = %.3f' % (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration

    # 循环结束后，计算每轮迭代的平均耗时mn和标准差sd, 最后将结果显示出来
    # 均方值 = 均值的平方 + 方差
    mn = total_duration/num_batches
    vr = total_duration_squared / num_batches - mn*mn  # 方差
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
          (datetime.now(), info_string, num_batches, mn, sd))


'''运算性能测试'''
batch_size = 16    # 一个批次的数据
num_batches = 100  # 测试一百个批次的数据
height, width = 299, 299  # 图片尺寸
inputs = tf.random_uniform((batch_size, height, width, 3))  # 生成随机图片数据作为input

# 使用slim.arg_scope加载前面定义好的inception_v3_arg_scope()，包含了各种默认参数
with slim.arg_scope(inception_v3_arg_scope()):
    # 调用inception_v3函数，传入inputs，获取logits和end_points
    logits, end_points = inception_v3(inputs, is_training=False)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# 测试forward性能
time_tensorflow_run(sess, logits, "Forward")






