"""
    作者：Heart Sea
    model： 5段卷积层，FC-4096, FC-4096, FC-1000, softmax
        输入层：image（224×224×3）
        第1段卷积： conv3-64，conv3-64，maxpool
        第2段卷积： conv3-128，conv3-128，maxpool
        第3段卷积： conv3-256，conv3-256，conv3-256，maxpool
        第4段卷积： conv3-512，conv3-512，conv3-512，maxpool
        第5段卷积： conv3-512，conv3-512，conv3-512，maxpool
        全连接     FC-4096
                  FC-4096
                  FC-1000
                  softmax

     各层tensor、memory、parameters：
        images     [32, 224, 224, 3]      memory：224×224×3=150k       参数：0

        conv1_1   [32, 224, 224, 64]      memory：224×224×64=3.2M      参数：(3×3×3)×64 = 1728
        conv1_2   [32, 224, 224, 64]      memory：224×224×64=3.2M      参数：(3×3×64)×64 = 36864
        pool1     [32, 112, 112, 64]      memory：112×112×64=800k      参数：0

        conv2_1   [32, 112, 112, 128]     memory：112×112×128=1.6M     参数：(3×3×64)×128 = 73728
        conv2_2   [32, 112, 112, 128]     memory：112×112×128=1.6M     参数：(3×3×128)×128 = 147456
        pool2     [32, 56, 56, 128]       memory：56×56×128=400K       参数：0

        conv3_1   [32, 56, 56, 256]       memory：56×56×256=800K       参数：(3×3×128)×256 = 294912
        conv3_2   [32, 56, 56, 256]       memory：56×56×256=800K       参数：(3×3×256)×256 = 589824
        conv3_3   [32, 56, 56, 256]       memory：56×56×256=800K       参数：(3×3×256)×256 = 589824
        pool3     [32, 28, 28, 256]       memory：28×28×256=200K       参数：0

        conv4_1   [32, 28, 28, 512]       memory：28×28×512=400K       参数：(3×3×256)×512 = 1179648
        conv4_2   [32, 28, 28, 512]       memory：28×28×512=400K       参数：(3×3×512)×512 = 2359296
        conv4_3   [32, 28, 28, 512]       memory：28×28×512=400K       参数：(3×3×512)×512 = 2359296
        pool4     [32, 14, 14, 512]       memory：14×14×512=100K       参数：0

        conv5_1   [32, 14, 14, 512]       memory：14×14×512=100K      参数：(3×3×512)×512 = 2359296
        conv5_2   [32, 14, 14, 512]       memory：14×14×512=100K      参数：(3×3×512)×512 = 2359296
        conv5_3   [32, 14, 14, 512]       memory：14×14×512=100K      参数：(3×3×512)×512 = 2359296
        pool5     [32, 7, 7, 512]         memory：7×7×512=25K         参数：0

        fc6   [32, 4096]                  memory：4096                参数：(7×7×512)×4096 = 102760448
        fc7   [32, 4096]                  memory：4096                参数：4096×4096 = 16777216
        fc8   [32, 1000]                  memory：4096                参数：4096×1000 = 4096000

    大部分参数在全连接，特别是第一个全连接参数个数最多。
    卷积层计算量大，耗时长
    数据集：不使用ImageNet数据集来训练， 构造出VGGNet-16网络结构，并评测前馈和反馈时间计算的耗时
    日期：12/04/2019
"""

from datetime import datetime
import math
import time
import tensorflow as tf
import tensorflow.contrib as ct


def print_activations(t):
    """
    :param t: tensor
    :return: 显示名称和尺寸
    """
    print(t.op.name, ' ', t.get_shape().as_list())


def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    """
    创建卷积层并把本层的参数存入列表

    :param input_op: 输入的Tensor
    :param name: 这一层的名称
    :param kh: kernel height, 即卷积核的高
    :param kw: kernel width,  即卷积核的宽
    :param n_out: 卷积核数量，即输出通道数
    :param dh: 步长的高
    :param dw: 步长的宽
    :param p: 参数列表
    :return: 返回卷积层的结果
    """
    # get_shape()[-1].value获取input_op的通道数，如输入图片尺寸224×224×3,中的那个3
    # input_op.get_shape() = TensorShape([Dimension(224), Dimension(224), Dimension(3)])
    n_in = input_op.get_shape()[-1].value    # input_op的输入通道数
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w",
                                 shape=[kh, kw, n_in, n_out],
                                 dtype=tf.float32,
                                 initializer=ct.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=tf.float32),
                             trainable=True,
                             name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]

        print_activations(activation)
        return activation


def fc_op(input_op, name, n_out, p):
    """
    定义全连接层
    :param input_op: 输入通道数
    :param name: 这一层的名称
    :param n_out: 输出通道数
    :param p: 参数
    :return: 返回全连接层的结果
    """
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w",
                                 shape=[n_in, n_out],
                                 dtype=tf.float32,
                                 initializer=ct.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b')
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)  # 相乘，相加，relu
        p += [kernel, biases]

        print_activations(activation)
        return activation


def mpool_op(input_op, name, kh, kw, dh, dw):
    """
    定义最大池化
    :param input_op: 输入通道数
    :param name: 这一层的名称
    :param kh: kernel height, 即池化的高
    :param kw: kernel width,  即池化的宽
    :param dh: 步长的高
    :param dw: 步长的宽
    :return: 返回最大池化层的结果
    """
    activation = tf.nn.max_pool(input_op,
                                ksize=[1, kh, kw, 1],
                                strides=[1, dh, dw, 1],
                                padding='SAME',
                                name=name)

    print_activations(activation)
    return activation


def inferece_op(input_op, keep_prob):
    """
    建立VGGNet-16网络结构
    :param input_op: 输入的Tensor
    :param keep_prob: 控制dropout比率的一个placeholder
    :return:
    """
    p = []  # 初始化参数列表
    print_activations(input_op)  # 224×224×3

    # 定义第1段卷积：2个卷积层，conv3-64, maxpool
    conv1_1 = conv_op(input_op, name="conv1_1", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)   # 224×224×64
    conv1_2 = conv_op(conv1_1, name="conv1_2", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)    # 224×224×64
    pool1 = mpool_op(conv1_2, name="pool1", kh=2, kw=2, dw=2, dh=2)                      # 112×112×64

    # 定义第2段卷积：2个卷积层，conv3-128，maxpool
    conv2_1 = conv_op(pool1, name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)     # 112×112×128
    conv2_2 = conv_op(conv2_1, name="conv2_2", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)   # 112×112×128
    pool2 = mpool_op(conv2_2, name="pool2", kh=2, kw=2, dw=2, dh=2)                      # 56×56×128

    # 定义第3段卷积：3个卷积层，conv3-256，maxpool
    conv3_1 = conv_op(pool2, name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)     # 56×56×256
    conv3_2 = conv_op(conv3_1, name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)   # 56×56×256
    conv3_3 = conv_op(conv3_2, name="conv3_3", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)   # 56×56×256
    pool3 = mpool_op(conv3_3, name="pool3", kh=2, kw=2, dw=2, dh=2)                      # 28×28×256

    # 定义第4段卷积：3个卷积层，conv3-512，maxpool
    conv4_1 = conv_op(pool3, name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)    # 28×28×512
    conv4_2 = conv_op(conv4_1, name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)  # 28×28×512
    conv4_3 = conv_op(conv4_2, name="conv4_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)  # 28×28×512
    pool4 = mpool_op(conv4_3, name="pool4", kh=2, kw=2, dw=2, dh=2)                     # 14×14×512

    # 规律：每一段卷积都会将图像的边长缩小到一半，但是卷积输出通道翻倍。
    #      这样图像面积缩小到1/4，输出通道变为2倍，因此输出tensor的总尺寸每次缩小一半。

    # 最后一段卷积维持通道数在512
    conv5_1 = conv_op(pool4, name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)    # 14×14×512
    conv5_2 = conv_op(conv5_1, name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)  # 14×14×512
    conv5_3 = conv_op(conv5_2, name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)  # 14×14×512
    pool5 = mpool_op(conv5_3, name="pool5", kh=2, kw=2, dw=2, dh=2)                     # 7×7×512

    # 将pool5的结果进行扁平化，展成7×7×512=25088的一维向量
    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name='resh1')

    # 连接一个隐含节点数4096的全连接层，激活函数为relu, dropout，训练时节点保留率为0.5，预测时为1
    fc6 = fc_op(resh1, name="fc6", n_out=4096, p=p)
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")

    # 完全与前面一样的全连接层
    fc7 = fc_op(fc6_drop, name="fc7", n_out=4096, p=p)
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")

    # 最后一个是1000个输出节点的全连接层，用softmax得到分类输出概率
    fc8 = fc_op(fc7_drop, name="fc8", n_out=1000, p=p)
    softmax = tf.nn.softmax(fc8)
    predictions = tf.argmax(softmax, 1)
    return predictions, softmax, fc8, p


def time_tensorflow_run(session, target, feed, info_string):
    """
    定义评测函数
    :param session: tensorflow 的session
    :param target:需要评测的运算算子，这里是 predictions
    :param feed: 预测时{keep_prob: 1.0}    训练时{keep_prob: 0.5}
    :param info_string: 测试的名称
    :return:
    """
    num_steps_burn_in = 10  # 预热轮数，只考虑10轮以后的计算时间
    total_dutation = 0.0
    total_dutation_squared = 0.0
    for i in range(num_steps_burn_in + num_batches):
        start = time.time()
        _ = session.run(target, feed_dict=feed)
        duration = time.time() - start
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' %
                      (datetime.now(), i-num_steps_burn_in, duration))
                total_dutation += duration
                total_dutation_squared += duration * duration
    mn = total_dutation / num_batches
    vr = total_dutation_squared / num_batches - mn*mn
    sd = math.sqrt(vr)
    print('%s: %s aross %d steps, %.3f +- %.3f sec / batch' %
          (datetime.now(), info_string, num_batches, mn, sd))


def run_benchmark():
    """
    主函数
    :return:评测forward 和 backward的运算性能，并不进行实质的训练和预测
    """
    with tf.Graph().as_default():   # 定义默认的Graph,方便后面使用
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size, image_size, image_size, 3],
                                              dtype=tf.float32,
                                              stddev=1e-1),
                             name="images")
        keep_prob = tf.placeholder(tf.float32)   # 控制dropout层的保留比率
        predictions, softmax, fc8, p = inferece_op(images, keep_prob)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        # forward计算的评测
        time_tensorflow_run(sess, predictions, {keep_prob: 1.0}, "forward")  # gpu:平均每轮耗时0.07s

        # backward训练过程的评测，所以要设置一个优化目标
        objective = tf.nn.l2_loss(fc8)
        grad = tf.gradients(objective, p)
        time_tensorflow_run(sess, grad, {keep_prob: 0.5}, "forward-backward")  # gpu:平均每轮耗时0.24s


batch_size = 2   # 如果使用较大的batch_size，gpu显存不够用
num_batches = 30
run_benchmark()






