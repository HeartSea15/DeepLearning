"""
    作者：Heart Sea
    功能：基于cifar10数据集实现图片的分类
    Model:
        输入层, 数据增强（翻转，剪切24*24，亮度，对比度），标准化

        卷积层, 卷积, 激活relu, 最大池化,
        卷积层, 卷积, 激活relu, LRN, 最大池化

        全连接, 激活relu，L2正则
        全连接, 激活relu，L2正则
        输出层, （全连接, 不需要l2，直接比较大小）

    日期：10/12/2019
"""
# 图像分类实验
# 下载Tensorflow Models库，以便使用CIFAR-10数据的类
# git clone https://github.com/tensorflow/models.git
# cd models/tutorials/image/cifar10

# 载入numpy和time，并载入CIFAR-10数据的类
import cifar10, cifar10_input
import tensorflow as tf
import numpy as np
import time
import math

max_steps = 3000    # 训练轮数
batch_size = 128
data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'     # 下载CIAR-10的默认路径


def variable_with_weight_loss(shape, stddev, w1):
    """
        对weight施加L2正则化处理，为了防止过拟合，正则化帮助我们惩罚权重特征，即特征的权重也会成为模型损失函数的一部分
        初始化weight函数，但是为了防止因为特征过多而引起的过拟合，给每个weight加一个L2的loss
        L1正则会造成稀疏的特征，大部分无用特征会被置为0。
        L2特征会让权重不过大，是权重比较平均.
        函数中用w1来控制L2 loss的大小
        使用tf.add_to_collection把weight_loss统一存放到一个collection,名字叫loss
    """
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))   # 从截断的正态分布中输出随机值
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var

# 使用CIFAR-10类下载数据集，并解压、展开到默认位置
cifar10.maybe_download_and_extract()

# 使用CIFAR-10类中的distorted_inputs函数产生训练使用的数据，并封装好作为tensor。
# 同时还进行了数据增强包括水平翻转、随机剪切、设置随机的亮度和对比度，以及对数据进行标准化。
# 数据增强采用16个线程进行加速,函数内部会产生线程池，需要使用时通过tensorflow queue进行调度
images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)

# 使用CIFAR-10类中的cifar10_input.inputs函数生成测试数据，
# 不需翻转、修改亮度,只需要剪切图片的中心24*24的大小的区块,标准化图片
images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

# 创建输入数据的placeholder，由于batch_size在之后定义网络结构会用到，所以数据尺寸中的第一个值即样本条数需要被预先设定，
# 数据尺寸中的图片尺寸为24*24,是剪切后的大小，通道数为3，代表图片是彩色的有RGB三条通道
image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])

"""
    构建第一个卷积层
    第一层卷积的权重不进行L2正则化，w1控制L2正则权重,
    利用tf.nn.conv2d对image_holder数据进行训练，
    然后经过relu非线性激活，之后最大池化，最后加入LRN层增加模型的泛化能力
"""
weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, w1=0.0)
kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))

# 最大池化的尺寸和步长不一致，这样可以增加数据的丰富性
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# 应用局部相应归一化层LRN，使得响应较大的值变得相对更大，并抑制其他反馈较小的神经元,增强了模型的泛化能力
# LRN对ReLU这种没有上限边界的激活函数比较有用，因为它会从附近的多个卷积核的响应中挑选比较大的反馈
# 但不适合sigmoid这种有固定边界并且能抑制过大值的激活函数
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 /9.0, beta=0.75)


"""
    构建第二个卷积层，相比于第一层，
    调换了最大池化层与LRN层的顺序，
    bias=0.1，而不是0,
    上一层的卷积核数量为64(即输出64个通道)，所以第三个维度(输入的通道数)为64
"""
weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, w1=0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

"""
    第一个全连接层，先把第二个卷积层的输出结果进行扁平化，然后使所有的参数都被L2正则约束，最后relu激活
    tf.reshape函数将每个样本都变成一维向量，
    get_shape函数获取数据扁平化后的长度
"""
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value
weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

"""
    第2个全连接层，隐含节点下降一半
"""
weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, w1=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

"""
   最后一层，不计入正则，
   不需要softmax就可以输出分类结果，直接计算模型inferfence输出结果，直接比较数值大小即可
"""
weight5 = variable_with_weight_loss(shape=[192, 10], stddev=1/192.0, w1=0.0)
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.add(tf.matmul(local4, weight5), bias5)
# 整个网络结构部分已经完成


def loss(logits, labels):
    """
        logits: 输出层的结果
        labels：实际结果

        计算CNN的loss，将softmax的计算和cross entropy loss的计算整合到一起
        使用tf.add_to_collection把cross entropy的loss添加到整体losses的collection中，
        使用tf.add_n将整体lossesDE collection中的全部loss求和，得到最终的loss，
        包括cross entropy loss,还有后面两个全连接层weight的L2 loss
    """
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('loss', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')
# 将logits节点和label_holder传入loss函数
loss = loss(logits, label_holder)

# 优化器选择Adam
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

# 使用tf.nn.in_top_k函数求出top1的准确率，输出分数最高的那一类的准确率
# 测试集用
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

# 创建默认的session，接着初始化全部模型参数
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# 启动前面图片数据增强的线程队列，共16个线程进行加速
tf.train.start_queue_runners()

"""
    开始正式训练，在每一个step的训练过程中，我们先使用session的run方法执行images_train、label_train的计算，
    获得一个batch的训练数据，再将这个batch的数据传入到train_op和loss的计算
    每隔10个step会计算并展示当前的loss，每秒钟能训练的样本数量，以及训练一个batch数据所花费的时间
"""
for step in range(max_steps):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])
    _,loss_value = sess.run([train_op, loss],
                            feed_dict={image_holder: image_batch, label_holder: label_batch})
    duration = time.time() - start_time
    if step % 10 == 0:
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)

        format_str=('step %d,loss=%.2f(%.1f examples/sec;%.3f sec/batch)')
        print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))

"""
    评测测试集的准确率
"""
num_examples = 10000      # 测试集样本数量

# 返回多少个batch
num_iter = int(math.ceil(num_examples / batch_size))    # math.ceil函数,返回数字的上入整数
true_count = 0
total_sample_count = num_iter * batch_size   # 总样本数
step = 0
while step < num_iter:
    image_batch, label_batch = sess.run([images_test, labels_test])
    predictions = sess.run([top_k_op],feed_dict={image_holder: image_batch,
                                                 label_holder: label_batch})
    true_count += np.sum(predictions)  # 汇总所有预测正确的结果
    step += 1
# 最后将评测结果计算出来
precision = true_count / total_sample_count
print('precision @ 1 = %.3f' % precision)
