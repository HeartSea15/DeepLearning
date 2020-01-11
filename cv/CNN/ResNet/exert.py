"""
    作者：Heart Sea
    实现的流程主要如下所示：

    0 导入相关模块
    1 定义Block类
    2 定义相关函数
            定义降采样方法、卷积操作方法、堆叠block方法、resnet的参数方法、残差学习单元（bottleneck）
    3 定义resnet_v2的实现方法
    4 定义不同深度的resnet_v2_50和resnet_v2_101的实现方法

    数据集：不使用数据集来训练， 只构造出ResNet V2网络结构，并评测前馈计算耗时
    日期：12/10/2019
    总结：gpu，forword平均每轮耗时0.12s, google inception v3为0.043，VGGNet的0.07s。
"""

import tensorflow as tf
from datetime import datetime
import math
import time
import collections
slim = tf.contrib.slim


# 定义Block类
class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    'A namee tuple describing a ResNet blick'
    '''一个典型的Block
        需要输入参数，分别是scope、unit_fn、args
        以Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)])为例，它可以定义一个典型的Block
        其中：
            1、block1就是这个Block的名称(或scope)
            2、bottleneck是ResNet V2中的残差学习单元
            3、[(256, 64, 1)] * 2 + [(256, 64, 2)]时这个Block的args，args是一个列表，其中每一个元素都对应一个bottleneck残差学习单元，
            前面两个都是(256, 64, 1)，最后一个是(256, 64, 2)。每个元素都是一个三元tuple，即(depth, depth_bottleneck, stride)
            比如(256, 64, 3)，代表构建的bottleneck残差学习单元(每个残差学习单元包含三个卷积层)中，第三层卷积输出通道数为256，
            前两层卷积输出通道数depth_bottleneck为64，且中间那层的步长stride为3
    '''


# 定义降采样方法
def subsample(inputs, factor, scope=None):
    """
    降采样
    :param inputs:
    :param factor: 采样因子
    :param scope:
    :return:
    """
    if factor == 1:
        return inputs

    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


# 定义卷积操作方法
def conv2d_same(inputs, num_outputs, kernel_size, stride, scope=None):
    """
        创建卷积层
    """
    # 判断步长是否为1，为1直接创建，padding为SAME
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, padding='SAME', scope=scope)

    # 不为1，显式的填充0，填充总数为kernel_size-1,
    else:
        # 修整Inputs,对inputs进行补零操作
        padding_total = kernel_size - 1  # pad zero的总数
        padding_beg = padding_total // 2
        padding_end = padding_total - padding_beg

        # pad(tensor, paddings, mode="CONSTANT", name=None, constant_value=0)：
        # 因为inputs是四维，因此paddings也是四维
        inputs = tf.pad(inputs, [[0, 0], [padding_beg, padding_end], [padding_beg, padding_end], [0, 0]])
        # 因为已经进行zero padding,所以只用一个VALID的padding的slim.conv2d来创建这个卷积层
        return slim.conv2d(inputs, num_outputs, kernel_size,
                           stride=stride, padding='VALID', scope=scope)


# 定义堆叠block方法
@slim.add_arg_scope
def stack_blocks_dense(net, blocks, outputs_collections=None):
    """
    定义堆叠block方法
    :param net: 输入
    :param blocks: 之前定义的Block的class的列表
    :param outputs_collections:收集各个end_points的collections
    :return:所有Block的所有Residual Unit都堆叠后，返回最后的net
    """

    for block in blocks:
        # 双层循环，遍历blocks，遍历res unit堆叠
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            # 用两个tf.variable_scope将残差学习单元命名为block1/unit_1的形式
            for i, unit in enumerate(block.args):
                with tf.variable_scope('unit_%d' % (i+1), values=[net]):
                    # 利用第二层循环拿到block中的args,将其展开为depth,depth_bottleneck,stride
                    unit_depth, unit_depth_bottleneck, unit_stride = unit

                    # 使用残差学习单元的生成函数unit_fn，顺序的创建并连接所有的残差学习单元
                    # block.unit_fn即调用了bottleneck()函数
                    net = block.unit_fn(net,
                                        depth=unit_depth,
                                        depth_bottleneck=unit_depth_bottleneck,
                                        stride=unit_stride)
                    # net = bottleneck(net,
                    #                 depth=unit_depth,
                    #                 depth_bottleneck=unit_depth_bottleneck,
                    #                 stride=unit_stride)

                    # 使用slim.utils.collect_named_outputs将输出net添加到collection中
                    net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)
    return net  # 返回最后的net作为函数结果


# 创建resnet通用的arg_scope
def resnet_arg_scope(is_training=True,
                     weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    """
    创建resnet通用的arg_scope
    :return:
    """
    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }
    # 通过slim.arg_scope将slim.conv2d的几个默认参数设置好
    with slim.arg_scope([slim.conv2d],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=slim.variance_scaling_initializer(),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,  # 标准化器设为BN
                        normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            # reset 原论文中的是padding='VALID',设为SAME，可以让特征对齐更简单
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc

# 定义核心的bottleneck残差学习单元
'''
知识点：
并不是所有的方法都能用arg_scope设置默认参数, 只有用@slim.add_arg_scope修饰过的方法才能使用arg_scope. 
例如conv2d方法, 它就是被修饰过的(见源码). 
所以, 要使slim.arg_scope正常运行起来, 需要两个步骤:
    1、用@add_arg_scope修饰目标函数
    2、用with arg_scope(...) 设置默认参数.
'''
@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, outputs_collections=None, scope=None):
    """
    定义核心的bottleneck残差学习单元,它和ResNet中的残差学习单元的主要区别是：
    1. 在每一层前都用了Batch Normalization,
    2. 对输入进行preactivation, 而不是在卷积进行激活函数处理
    :param inputs:输入
    :param depth: 输出通道数
    :param depth_bottleneck:前两层卷积输出通道数
    :param stride:中间那层卷积层的步长
    :param outputs_collections:收集end_points的collection
    :param scope:这个unit的名称
    :return:
    """
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        # 获取输入的最后一个维度，即输出通道数。  min_rank=4限定最少为4个维度
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)

        # 对输入进行BN(Batch Normalization)操作,并使用ReLU函数进行预激活Preactivate
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')

        '''定义shortcut，即旁路的弯曲的支线'''
        # 如果残差单元的输入通道depth_in与第三层卷积的输出通道depth一致
        if depth == depth_in:
            # 使用subsample按步长为stride对inputs进行空间降采样
            # (因为输出通道一致了，还要确保空间尺寸和残差一致，因为残差中间那层的卷积步长为stride，tensor尺寸可能会缩小)
            shortcut = subsample(inputs, stride, 'shortcut')
        else:  # 输出通道不一致
            # 使用1×1的卷积核改变其通道数，并使用与步长为stride确保空间尺寸与残差一致
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride, normalizer_fn=None,
                                   activation_fn=None, scope='shortcut')

        '''残差residual,三层卷积'''
        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1, scope='conv1')
        residual = conv2d_same(residual, depth_bottleneck, 3, stride, scope='conv2')  # 步长为stride，并进行补零操作
        residual = slim.conv2d(residual, depth, [1, 1], stride=1, normalizer_fn=None,
                               activation_fn=None, scope='conv3')  # 最后一层卷积没有正则项也没有激活函数
        output = shortcut + residual
        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)  # 将结果添加进collection并返回


# 生成ResNet的主函数：卷积，池化，残差模块组（转stack_blocks_dense函数），全局平均池化，卷积，softmax
def resnet_v2(inputs,
              blocks,
              num_classes=None,
              global_pool=True,
              include_root_block=True,
              reuse=None,
              scope=None):
    """
    生成ResNet的主函数：卷积，池化，残差模块组（转stack_blocks_dense函数），全局平均池化，卷积，softmax
    只要预先定义好网络的残差学习模块组blocks，它就可以生成对应的完整的ResNet
    :param inputs: 输入
    :param blocks: 定义好的Block类列表
    :param num_classes: 最后输出的类数
    :param global_pool: 是否加上最后一层全局平均池化
    :param include_root_block: 是否加上ResNet网络最前面通常使用的7*7卷积和最大池化
    :param reuse: 是否重用
    :param scope: 整个网络的名称
    :return:
    """

    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:

        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck, stack_blocks_dense],
                            outputs_collections=end_points_collection):

            net = inputs

            # 根据include_root_block标记，创建ResNet最前面的64输出通道的步长为2的7*7卷积
            if include_root_block:
                with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
                    net = conv2d_same(net, 64, 7, stride=2, scope='conv1')

                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
            # 经过两个步长为2的层，图片尺寸已经缩小为1/4.

            # 使用前面定义好的函数stack_blocks_dense将残差学习模块组生成好，得到其输出
            net = stack_blocks_dense(net, blocks)
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')

            # 根据标记添加全局平均池化层tf.reduce_mean，效率比avg_pool高
            if global_pool:
                net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)

            # 根据是否有分类数，添加一个1*1卷积
            if num_classes is not None:
                net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits')

            # 通过该方法将collection转化为python的dict词典
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

            # 添加Softmax层输出网络结果
            if num_classes is not None:
                end_points['predictions'] = slim.softmax(net, scope='predictions')

            return net, end_points


# 设计层数分别为50,101,152,200的resnet
def resnet_v2_50(inputs,
                  num_classes=None,
                  global_pool=True,
                  reuse=None,
                  scope='resnet_v2_50'):
    """
    设计层数为50层的ResNet
    """
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)]

    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)


def resnet_v2_101(inputs,
                  num_classes=None,
                  global_pool=True,
                  reuse=None,
                  scope='resnet_v2_101'):
    """
    设计层数为101层的ResNet,Resnet不断使用步长为2的层来缩减尺寸，同时输出通道数也在持续增加
    """
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)]

    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)


def resnet_v2_152(inputs,
                  num_classes=None,
                  global_pool=True,
                  reuse=None,
                  scope='resnet_v2_152'):
    """
    设计层数为152层的ResNet,Resnet不断使用步长为2的层来缩减尺寸，同时输出通道数也在持续增加
    :param inputs:
    :param num_classes:
    :param global_pool: 是否加上最后一层全局平均池化
    :param reuse:
    :param scope:
    :return:
    """
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)]

    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)


def resnet_v2_200(inputs,
                  num_classes=None,
                  global_pool=True,
                  reuse=None,
                  scope='resnet_v2_200'):
    """
    设计层数为200层的ResNet,Resnet不断使用步长为2的层来缩减尺寸，同时输出通道数也在持续增加
    :param inputs:
    :param num_classes:
    :param global_pool: 是否加上最后一层全局平均池化
    :param reuse:
    :param scope:
    :return:
    """
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 23 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)]

    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)


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


if __name__ == '__main__':
    batch_size = 32
    height, width = 224, 224
    num_batches = 100
    inputs = tf.random_uniform((batch_size, height, width, 3))

    with slim.arg_scope(resnet_arg_scope(is_training=False)):
        net, end_points = resnet_v2_152(inputs, 1000)    # 测试ILSVRC 2015冠军的版本152

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    time_tensorflow_run(sess, net, "Forward")