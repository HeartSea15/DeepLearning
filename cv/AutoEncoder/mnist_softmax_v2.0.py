"""
    作者：Heart Sea
    功能：实现softmax regression 识别手写数字
    Model: 一层全连接,然后softmax,取概率值最大的那个
    2.0：数据集位置与执行文件在同一个文件夹中
    版本：2.0
    日期：10/10/2019
"""

import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("D:/software/pycharm/shizhan/softmax/MNIST_data/", one_hot=True)

print(mnist.train.images.shape, mnist.train.labels.shape)  # 训练集
print(mnist.test.images.shape, mnist.test.labels.shape)    # 测试集
print(mnist.validation.images.shape, mnist.validation.labels.shape)  # 验证集
# (55000, 784) (55000, 10)
# (10000, 784) (10000, 10)
# (5000, 784) (5000, 10)

import tensorflow as tf
x = tf.placeholder(tf.float32, [None, 784])  # 创建输入数据的地方，数据类型float32，数据尺寸[None, 784]。None表示不限条数的输入，784是每条输入是一个784维的向量
W = tf.Variable(tf.zeros([784, 10]))    # 创建权值参数矩阵，尺寸[784, 10]
b = tf.Variable(tf.zeros([10]))         # 创建bias参数向量，尺寸[10],python执行结果是一行10列,matlab执行结果是10行10列
y = tf.nn.softmax(tf.matmul(x, W) + b)  # 进行Softmax Regression算法，y是预测的概率分布,y的shape为（None, 10）
# softmax是tf.nn下面的一个函数，而tf.nn则包含了大量神经网络的组件。

# 训练模型
# 对多分类问题，经常使用交叉熵作为loss function
# softmax的交叉熵公式：对所有样本的交叉熵损失求和再平均，再负
# 计算交叉熵，判断模型对真实概率分布估计的准确程度
# y_ * tf.log(y)维度都是[None, 10]，因此两者相乘（不是矩阵相乘），实质是对应行相乘
# 用 tf.reduce_sum 根据 reduction_indices=[1] 指定的参数计算y中第二个维度所有元素的总和（10个类别求和）,
# tf.reduce_mean用来对每个batch数据求均值
y_ = tf.placeholder(tf.float32, [None, 10])  # 定义placeholder，y_是真实的概率分布
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# 选择随机梯度下降SGD以0.5的学习速率最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
# 上面三行可改写为下面一行
# tf.global_variables_initiallizer().run()

for i in range(1000):      # 模型循环训练1000次,从0开始，999结束
    batch_xs, batch_ys = mnist.train.next_batch(100)    # 随机抓取训练数据中的100个批处理数据点
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
# 完成模型训练

# 评估模型
# tf.argmax是从一个tensor中寻找最大值的序号,tf.argmax(y,1)求预测数字中概率最大的那一个，tf.argmax(y_,1)求样本的真实数字类别
# tf.equal 判断预测的数字类别是否就是正确的类别
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# 用tf.cast将之前的correct_prediction输出的bool值转换为float32，再求平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))   #输入数据，计算准确率
# 0.9192