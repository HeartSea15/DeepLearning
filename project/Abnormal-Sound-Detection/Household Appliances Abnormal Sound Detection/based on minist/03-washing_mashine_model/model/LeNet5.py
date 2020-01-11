"""
    作者：Heart Sea
    模型：LeNet5
    功能：基于minist, CNN实现识别手写数字
    Model:
        输入层, 1D转换2D
        卷积层, 卷积, 激活relu, 最大池化
        卷积层, 卷积, 激活relu, 最大池化
        全连接, 2D转换1D, 激活relu
        Dropout层,
        全连接, softmax
        输出层
    版本：3.0
    日期：10/11/2019

    LeNet5当时的特性有如下几点：
    （1）每个卷积层包含三个部分：卷积、池化和非线性激活函数
    （2）使用卷积提取空间特征
    （3）降采样（subsample）的平均池化层（average pooling）
    （4）双曲正切（tanh）或S型（sigmoid）的激活函数
    （5）MLP作为最后的分类器
    （6）层与层之间的稀疏连接减少计算复杂度
"""


from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()


# 由于使用ReLU
# 需要使用正态分布给参数加噪声（这里加入截断的正态分布噪声），来打破完全对称并且避免0梯度
# 还需要给偏置赋值小的非零值来避免死亡神经元
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 定义卷积层函数，strides代表卷积模板移动的步长，都是1表示划过图片每一个点
# padding表示边界处理方式，SAME让卷积的输入和输出保持同样的尺寸
# x是输入，w卷积的参数
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 定义最大池化函数，ksize参数是滤波器大小，表示2*2的滤波器，
# strides设为横竖两个方向以2为步长，步长如果为1，得到一个尺寸不变的图片
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# CNN会利用到空间的结构信息，需要将1D转换成2D
# 利用tf.reshape函数对输入的一维向量还原为28x28的结构，-1代表样本数量不固定，最后1代表颜色通道数量
# x特征，y_真实的label
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 定义第一个卷积层，[5, 5, 1, 32]代表 卷积核尺寸为5x5,1个通道,32个不同卷积核
# 对权重、偏置初始化，然后经卷积层和激活函数激活，最后池化操作
# h_pool1尺寸：14*14*32
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 定义第二个卷积层，64是卷积核的数量，提取64种特征
# h_pool2尺寸：7*7*64（经过两次池化）
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
# 经过两层池化，边长变为7*7，所以第二个卷积层输出的tensor尺寸为7*7*64

# 全连接层处理
# h_fc1尺寸：1*1024
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 减轻过拟合，使用Dropout层
# h_fc1尺寸：1*1024
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 建立Softmax层与Dropout层连接，最后输出概率
# y_conv尺寸：1*10
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 定义损失函数，指定Adam优化器优化
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 计算分类准确率
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.global_variables_initializer().run()

# 开始训练,训练2万次，mini_batch为50，每100次显示分类精度
# 评测时，keep_prob设为1，用以实时监测模型性能
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# 对测试集上进行测试
print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
# 99.2%