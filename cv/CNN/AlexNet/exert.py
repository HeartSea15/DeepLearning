"""
    作者：Heart Sea
    model：
        输入层：image
        第1个卷积层：卷积，relu, lrn, 最大池化
        第2个卷积层，卷积，relu, lrn, 最大池化
        第3个卷积层，卷积，relu
        第4个卷积层，卷积，relu
        第5个卷积层，卷积，relu，最大池化
    数据集：不使用ImageNet数据集来训练， 只使用随机图片数据测试前馈和反馈时间计算的耗时

    日期：12/03/2019
"""
from datetime import datetime
import math
import time
import tensorflow as tf

batch_size = 32
num_batches = 20  # 50个batch


def print_activations(t):
    """
    :param t: tensor
    :return: 显示名称和尺寸
    """
    print(t.op.name, ' ', t.get_shape().as_list())


# 设计AlexNet结构
def inference(images):
    """
    :param images:[batch, in_height, in_width, in_channels]
    :return: 最后一层pool5（第5个池化层）及parameters（AlexNet中所有需要训练的模型参数）
    """
    parameters = []

    # 定义第1个卷积层
    with tf.name_scope('conv1') as scope:
        # 初始化卷积核的参数
        kernel = tf.Variable(tf.random.truncated_normal([11, 11, 3, 64], dtype=tf.float32, stddev=0.1),
                             name='weight')
        # 对输入images进行卷积，步长4（横向、纵向间隔都是4）
        # 计算tensor维度时，shape = (M-m+2p)/s+1，不能整除，向下取整
        # padding='SAME'是等宽卷积，p = (m-1)/2
        # padding='VALID'是窄卷积，p = 0
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        # 卷积层的biase全部初始化为0
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True,
                             name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print_activations(conv1)
        parameters += [kernel, biases]
    # 在第一个卷积层后再添加LRN层和最大池化层
    # lrn1中的参数是AlexNet论文中的推荐值。
    # 不过除了AlexNet，其他的经典CNN基本都不采用，LRN层效果不明显，而且会让forward和backwood的速度大大下降
    lrn1 = tf.nn.lrn(conv1, depth_radius=4, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn1')
    # padding='VALID',取样时不能超过边框
    pool1 = tf.nn.max_pool2d(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
    print_activations(pool1)

    # 定义第2个卷积层，大部分和第一个卷积层相同，只有几个参数不同。
    # 区别在于：卷积核尺寸5*5，输入通道数64，卷积核数量192,。卷积步长全部设为1，即扫描全图像素
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.random.truncated_normal([5, 5, 64, 192], dtype=tf.float32, stddev=1e-1),
                             name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                             trainable=True,
                             name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        print_activations(conv2)
        parameters += [kernel, biases]
    # 在第2个卷积层后再添加LRN层和最大池化层 ,和第个卷积层后的操作完全一样
    lrn2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn2')
    pool2 = tf.nn.max_pool2d(lrn2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='pool2')
    print_activations(pool2)

    # 定义第3个卷积层，基本结构和前面两层相同，只有几个参数不同，卷积层后面没有LRN和最大池化。
    # 区别在于：卷积核尺寸3*3，输入通道数192，卷积核数量384,。卷积步长全部设为1，即扫描全图像素
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.random.truncated_normal([3,3,192,384], dtype=tf.float32, stddev=1e-1),
                             name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1,1,1,1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True,
                             name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        print_activations(conv3)
        parameters += [kernel, biases]

    # 定义第4个卷积层
    # 区别：卷积核尺寸3*3，输入通道数384，卷积核数量降为256
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.random.truncated_normal([3,3,384,256], dtype=tf.float32, stddev=1e-1),
                             name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1,1,1,1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True,
                             name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        print_activations(conv4)
        parameters += [kernel, biases]

    # 定义第5个卷积层
    # 卷积核尺寸3*3，输入通道数256，卷积核数量也为256.
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.random.truncated_normal([3,3,256,256], dtype=tf.float32, stddev=1e-1),
                             name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1,1,1,1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True,
                             name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        print_activations(conv5)
        parameters += [kernel, biases]
    # 添加最大池化
    pool5 = tf.nn.max_pool2d(conv5, ksize=[1, 3, 3, 1], strides=[1,2,2,1], padding='VALID', name='pool5')
    print_activations(pool5)
    return pool5, parameters
# 正式使用AlexNet进行训练和评测时，需要添加3个全连接层，隐含节点分别是4096,4096,1000
# 这里只进行速度评测，由于3个全连接的计算量很小，对计算耗时影响非常小，就没放到速度评测中


def time_tensorflow_run(session, target, info_string):
    """
    :param session: tensorflow 的session
    :param target: 需要评测的运算算子，这里是卷积网络最后一个池化层的输出pool5
    :param info_string: 测试的名称
    :return: 评估每轮迭代耗时
    """
    num_steps_burn_in = 10  # 预热轮数，只考两10轮以后的计算时间
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


def run_benchmark():
    """
    主函数
    :return:
    """
    with tf.Graph().as_default():  # 定义默认的Graph,方便后面使用
        image_size = 224
        # 由于只使用随机图片数据测试前馈和反馈时间计算的耗时，这里用正太分布的随机Tensor
        images = tf.Variable(tf.random.normal([batch_size, image_size, image_size, 3],
                                              dtype=tf.float32,
                                              stddev=1e-1))
        pool5, parameters = inference(images)

        init = tf.compat.v1.global_variables_initializer()  # tf.global_variables_initializer的升级
        sess = tf.compat.v1.Session()  # tf.Session()的升级
        sess.run(init)

        # forward计算的评测
        time_tensorflow_run(sess, pool5, "forward")  # cpu:平均每轮耗时3.133s，gpu:0.01s

        # backward训练过程的评测，所以要设置一个优化目标
        objective = tf.nn.l2_loss(pool5)
        grad = tf.gradients(objective, parameters)
        time_tensorflow_run(sess, grad, "forward-backward")  # cpu:平均每轮耗时21.529s,  gpu:0.03s


run_benchmark()



