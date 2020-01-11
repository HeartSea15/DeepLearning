"""
    作者：Heart Sea
    功能：tessorflow实现多层感知机/多层神经网络 Multi-Layer Perceptron, MLP
        加入：减轻过拟合的Dropout, 自适应学习速率的Adagrad, 可解决梯度弥散的激活函数ReLU
    Model:
        1个输入层
        全连接
        1个隐含层, 激活ReLU, Dropout
        全连接
        1个输出层, softmax
    版本：2.0
    日期：10/10/2019
"""

# 载入tensorflow加载数据集mnist
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 创建默认的InteractiveSession,后面各项操作无须指定session了
sess = tf.InteractiveSession()

in_units = 784        # 输入层节点数
h1_units = 300        # 隐含层节点数

# W1，b1为隐含层的权重和偏置
# W2，b2为输出层的权重和偏置，全部置为0，因为对于sigmoid，在0处最敏感，梯度最大
# 初始化参数W1为截断的正态分布，标准差为0.1
# 由于使用ReLU，需要使用正态分布给参数加噪声，来打破完全对称并且避免0梯度
# 在其他的一些模型中，还需要给偏置赋值小的非零值来避免死亡神经元
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))

# 定义输入x的placeholder，由于Dropout的比率keep_prob在测试和训练时不一样，训练时小于1，预测时大于1
# keep_prob训练时小于1，用以制造随机性，防止过拟合；预测时大于1，即使用全部特征来预测样本的类别
# 所以也把Dropout的比率keep_prob作为计算图的输入，并定义成一个placeholder
x = tf.placeholder(tf.float32, [None, in_units])
keep_prob = tf.placeholder(tf.float32)      # 保留节点的概率(保留数据而不置0的比例)

# 建立隐藏层和输出层，并且调用Dropout函数处理隐含层输出数据
# ReLU的优点：单侧抑制，相对宽阔的兴奋边界，稀疏激活性
# 隐含层的激活函数用ReLU可以提高训练速度和模型准确率
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)          # 隐含层激活函数为ReLU
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)     # 实现Dropout的功能，即随机将一部分节点置为0
y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)  # 输出层，shape和y_一样，[None, 10]

# 定义损失函数cross_entropy，并指定自适应优化器Adagrad优化损失函数
# y_和y是相同的维度[None, 10]，两者相乘，实质求内积，求和按照[1]列求和（一个样本的所有类别求和），求和后为一维向量，平均后为1个数
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

tf.global_variables_initializer().run()

# 由于加入隐含层，需要更多的训练迭代来优化模型参数达到一个比较好的效果
# 进行训练3000个batch，每个batch有100个样本，一共30万的样本，相当于对全数据进行5轮迭代
for i in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})

# 对模型进行准确性评测，其中加入了keep_prob=1
# tf.cast是将correct_prediction输出的bool值转换为float32
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
# 0.9778