"""
    作者：Heart Sea
    功能：tessorflow实现去噪自编码器Additive Gaussian Noise,AGN
    Model:
        1个输入层, 加噪声
        全连接
        1个隐含层, 激活
        全连接
        1个输出层
    版本：1.0
    日期：10/10/2019
"""

import numpy as np                       # 导入常用库numpy
import sklearn.preprocessing as prep     # 导入sklearn中的preprocessing模块，负责预处理数据标准化
import tensorflow as tf                  # 导入tensorflow
from tensorflow.examples.tutorials.mnist import input_data    # 导入mnist数据集


# 实现标准均匀分布的Xaiver初始化器(根据某一层网络的输入、输出节点数量自动调整最合适的权重分布)
# Xaiver就是让权值满足：均值=0，方差=2/(n_in+n_out)
# 分布可以用均匀分布或者高斯分布,这里采用均匀分布，方差=2/(n_in+n_out)=（max-min）^2/12
def xavier_init_(fan_in, fan_out, constant=1):  # xavier_init_(self.n_input, self.n_hidden)
    """
    :param fan_in: 输入节点的数量, 行数
    :param fan_out: 输出节点的数量, 列数
    :param constant: 1
    :return: 返回一个比较适合softplus等激活函数的权重初始分布w1
    """
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


# 定义一个去噪自编码的class，此类包括一个构建函数_init_(),还有一些成员函数
class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdadeltaOptimizer(), scale=0.1):
        """
        :param n_input: 输入变量数
        :param n_hidden: 隐含层节点数
        :param transfer_function: 隐含层激活函数，默认为softplus
        :param optimizer: 默认为Adam
        :param scale: 高斯噪声系数,默认0.1
        """
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()  # 使用_initialize_weights()函数初始化参数,后面会定义
        self.weights = network_weights

    # 定义网格结构：输入层，隐含层，输出层
        # x每行是一个样本，列是特征
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        # 建立能提取特征的隐含层，給输入添加维度为(n_input,)的正态分布的噪声
        # tf.random_normal((n_input,))里的小括号可换成中括号，一行n_input列个正态随机个数，没有设置随机种子
        # self.x + scale * tf.random_normal((n_input,))相当于每个样本的n_input个特征都加了噪声
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)),
                                                     self.weights['w1']),
                                           self.weights['b1']))
        # 建立输出层
        self.reconstruction = tf.add(tf.matmul(self.hidden,
                                               self.weights['w2']),
                                     self.weights['b2'])

    # 定义自编码器的损失函数，用平方误差作为cost，输出与输入之差再平方，求和，0.5
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(
            self.reconstruction, self.x), 2.0))         # 使用平方误差作为cost
        self.optimizer = optimizer.minimize(self.cost)  # 优化器为损失进行优化

        init = tf.global_variables_initializer()        # 初始化全部模型参数
        self.sess = tf.Session()                        # 建立Session
        self.sess.run(init)

    # 定义参数初始化函数_initialize_weights
    # 输出层权重和偏置不含激活函数，直接初始化为0
    def _initialize_weights(self):
        all_weights = dict()  # 创建字典dict
        all_weights['w1'] = tf.Variable(xavier_init_(self.n_input, self.n_hidden))   # 返回一个比较适合softplus等激活函数的权重初始分布
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, X):
        """
        aim: 定义计算损失cost及执行一步训练的函数partial_fit
        :param X: 一个batch数据
        :return: 当前损失
        """
        cost, opt = self.sess.run((self.cost, self.optimizer),
                                  feed_dict={self.x: X, self.scale: self.training_scale})
        return cost

    def calc_total_cost(self, X):
        """
        aim: 定义一个只计算cost的函数，主要是在训练完毕后，在测试集上对模型性能进行测评，不会触发训练操作
        :param X: 测试集数据
        :return: 平方误差cost
        """
        return self.sess.run(self.cost, feed_dict={self.x: X,
                                                   self.scale: self.training_scale})

    def transform(self, X):
        """
        aim: 定义transform函数（自编码器的前半部分），目的是提供一个接口来获取高阶特征
        :param X:
        :return: 返回自编码器隐含层的输出结果 hidden
        """
        return self.sess.run(self.hidden, feed_dict={self.x: X,
                                                     self.scale: self.training_scale})

    def generate(self, hidden=None):
        """
        定义generate函数（自编码器的后半部分），将transform提取的高阶特征经过重建层复原为原始数据
        :param hidden: 隐含层的输出结果
        :return: 高阶特征经过重建层复原为原始数据
        """
        if hidden is None:
            hidden = np.random.normal(size=self.weights["b1"])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        """
        整体运行transform和generate两个过程
        :param X: 原始数据
        :return: 复原的原始数据
        """
        return self.sess.run(self.reconstruction, feed_dict={self.x: X,
                                                             self.scale: self.training_scale})

    # 定义getWeights函数，获取隐含层的权重W1
    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    # 定义getBiases函数，获取隐含层偏置系数b1
    def getBiases(self):
        return self.sess.run(self.weights['b1'])

# 去噪自编码器的类已经定义完，包括神经网络的设计、权重的初始化


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)   # 载入minist数据集


# 定义standard_scale函数，对训练、测试数据标准化，
# 标准化即让数据变成0均值，且标准差为1的分布。方法是先减去均值，再除以标准差
# 为保证训练、测试数据使用完全相同的Scaler，需要先在训练数据上fit出一个共用的Scaler，再将这个Scaler用到训练数据和测试数据上
def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


# 定义一个随机取block数据的函数
# random.randint(a, b)，用于生成一个指定范围a <= n <= b的整数
# 将这个随机数作为block的起始位置，顺序取到一个batch size的数据。这属于不放回原样，可以提高数据利用率
def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]


# 使用standard_index函数标准化训练集，测试集
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

# 定义常用参数
n_samples = int(mnist.train.num_examples)  # 总训练样本数
training_epochs = 20                       # 最大训练轮数
batch_size = 128                           # 批数据，即多少个样本作为一个batch
display_step = 1                           # 每隔一轮epoch显示一次损失cost

# 创建一个AGN自编码器的实例，并设置输入层节点数、隐含层节点数、隐含层激活函数、优化器、噪声系数
autoencoder = AdditiveGaussianNoiseAutoencoder(n_input=784,
                                               n_hidden=200,
                                               transfer_function=tf.nn.softplus,
                                               optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                               scale=0.01)

# 开始训练
for epoch in range(training_epochs):                # 迭代训练轮数
    # print(epoch)
    avg_cost = 0.                                   # 每一轮开始时，平均损失=0
    total_batch = int(n_samples / batch_size)       # 样本总数/batch大小=需要的batch数
    for i in range(total_batch):                    # 迭代一轮中所有的批数据
        # print(i)
        batch_xs = get_random_block_from_data(X_train, batch_size)       # 随机抽取block数据

        cost = autoencoder.partial_fit(batch_xs)      # 使用partial_fit训练这个当前的batch数据的平方误差
        avg_cost += cost / n_samples * batch_size     # 将当前的cost整合到avg_cost，计算平均cost

# 每一轮迭代后，显示当前的迭代数和这一轮迭代的acg_cost
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1),
              "cost =", "{:.9f}".format(avg_cost))

# 对测试集进行测试，评价指标仍然是平方误差
print("Total cost: " + str(autoencoder.calc_total_cost(X_test)))
