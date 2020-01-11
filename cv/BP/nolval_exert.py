"""
    作者：Heart Sea
    功能：np实现单隐层的BP算法，没有用到验证集，正则化
    Model:
        1个输入层
        全连接
        1个隐含层, 激活ReLU
        全连接
        1个输出层, 激活ReLU，损失函数为平方损失

    测试集准确率95.14%

    版本：3.0
    日期：10/19/2019
"""

import gzip
import pickle
import random
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt


# 加载MNIST数据集
def load_data():
    f = gzip.open('./data/mnist.pkl.gz')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return training_data, validation_data, test_data
training_data, validation_data, test_data = load_data()

print(training_data[0].shape, training_data[1].shape)
print(validation_data[0].shape, validation_data[1].shape)
print(test_data[0].shape, test_data[1].shape)
# (50000, 784) (50000,)
# (10000, 784) (10000,)
# (10000, 784) (10000,)

# image = training_data[0][0].reshape(28, 28)    # 训练集label的第一个图片，每个样本（图片）28行，28列，即784维特征
# plt.imshow(image)
# plt.show()

print(training_data[1][0])   # 训练集label的第一个数字5


#----------------构建支持向量机SVM模型，作为人工神经网络模型的性能对比的基模型-------

# 以svm模型为基模型（也可以用其他模型），神经网络不是号称学习能力强吗，准确率必须高于基模型。
def svm_baseline():
    training_data, validation_data, test_data = load_data()

    clf = svm.SVC()
    clf.fit(training_data[0][0:10000], training_data[1][0:10000])
    # 可以全部训练5万条，这里只设置10000条数据进行训练，有gpu可以全部训练

    predictions = [int(a) for a in clf.predict(test_data[0])]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
    # zip后，两两对应一个元组，返回的是一个返回的是一个对象。如需展示列表，需手动 list() 转换，或者for
    # int(True)=1, int(False)=0,便于后面sum计算求和

    print("Baseline classifier using an SVM.")
    print(str(num_correct) + " of " + str(len(test_data[1])) + " values correct.")
#     10000条数据，9214条分类正确，正确率92.14%，如果用全部数据训练，94.35%

svm_baseline()


#----------------- 封装训练数据集、验证数据集和测试数据集 --------------
# 只有训练集需要把标签one-hot编码
def load_data_wrapper():
    tr_d, va_d, te_d = load_data()

    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]  # 训练集的每一个样本转换成784行，1列的列表，列表里面是数组格式
    training_results = [vectorized_result(y) for y in tr_d[1]]  # 训练集的每一个标签one-hot编码

    training_data = zip(training_inputs, training_results)  # 封装训练集的样本和标签

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])  # 封装验证集的样本和标签

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])  # 封装测试集的样本和标签

    return training_data, validation_data, test_data


def vectorized_result(j):
    e = np.zeros((10, 1))  # 10行1列的二维数组
    e[j] = 1.0
    return e


training_data, validation_data, test_data = load_data_wrapper()  # 三个数据集都是zip格式


#------------------------- 定义前馈神经网络模型 ------------------------------------
class Network(object):

    def __init__(self, sizes):  # [784, 30, 10]
        self.num_layers = len(sizes)
        self.sizes = sizes

        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # biases[0]为第二层的初始化偏置  (30, 1)
        # biases[1]为第三层的初始化偏置   (10, 1)

        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        # self.weights[0]第二层的初始化权重，(30, 784)
        # self.weights[1]为第三层的初始化权重，(10, 30)

    def feedforward(self, a):
        """
            输入：测试集的实例
            输出：前向传播求a3(预测输出y_hat)
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):  # training_data, 30, 10, 3.0, test_data
        """
            定义批次随机梯度下降
            epochs：30个循环
            mini_batch_size：每组10个样本
            eta：3.0 学习率
            test_data: zip格式的测试样本
        """
        training_data = list(training_data)  # zip解锁为list形式，里面是元祖存放，一个样本对一个标签
        n = len(training_data)  # 5万个
        # print(n)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)  # 1万个

        for j in range(epochs):  # 30次循环
            random.shuffle(training_data)  # 打乱训练集样本的顺序

            # n=5万个数据集,
            # mini_batch_size(10个样本作为一个一组)，一共有n/mini_batch_size=5万/10=5k组,每组10个样本
            mini_batches = [training_data[k: k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            # print(len(mini_batches))   # 5k组

            # 每一组mini_batch10个样本
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)  # 每一组10个样本更新当前的权重和偏置
            print("Epoch %s training complete" % j)

            # 测试数据，边训练边评估
            if test_data:
                print("Epoch {} : {} / {}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """
            输入：mini_batch = 10个样本
                  eta = 3.0 学习率

            输出：每一组mini_batch更新当前的权重和偏置 ，即每10个样本更新一次b和w

        """

        nabla_b = [np.zeros(b.shape) for b in self.biases]  # 依次是第二层，第三层的偏置
        nabla_w = [np.zeros(w.shape) for w in self.weights]  # 依次是第二层，第三层的权重

        for x, y in mini_batch:
            # 对10个样本，取每个样本的x特征向量, y标签（one-hot）
            # x为784行，1列的向量，y为10行一列的标签
            # 依次对10个样本的两层实现损失函数在各层的偏置梯度累加、权重梯度累加

            delta_nabla_b, delta_nabla_w = self.backprop(x, y)  # 损失函数对每一个样本各层的关于b、w的梯度

            # for i,j in zip([2,3],[2,3])
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]  # 10个样本，关于偏置梯度累加
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]  # 10个样本，关于权重梯度累加

        #  更新当前的权重和偏置
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]

    def backprop(self, x, y):
        """
            前向传播计算
            误差的反向传播计算
            输入：
                x 训练实例的每一个样本，784*1
                y 标签，10*1
            输出： 损失函数对每一个样本各层的关于b、w的梯度
        """
        # 占位
        nabla_b = [np.zeros(b.shape) for b in self.biases]  # 依次是第二层，第三层的偏置
        nabla_w = [np.zeros(w.shape) for w in self.weights]  # 依次是第二层，第三层的权重

        activation = x     # 样本的784个维度，784*1
        activations = [x]  # 分层存储每层的输出项（z激活）,a向量, 依次是第一层a1（784*1）, 第二层a2（30*1）， 第三层a3（10*1）
        zs = []            # 分层存储每层的 z 向量,  依次是第二层z2（30*1），第三层z3（10*1）

        # 前向计算
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b  # 第2层维度：30*1，第3层维度：10*1, numpy类型
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        #  第3层误差：delta3（列向量与列向量的内积），10*1
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])

        nabla_b[-1] = delta  # 损失函数在第三层关于偏置的梯度，为第三层的误差，列向量, 10*1
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # 关于权重的梯度，10*30，损失函数在本层关于权值的梯度=本层误差*上一层的激活输出

        #  第2层误差：delta2，30*1
        for m in range(2, self.num_layers):  # 这里只有三层，所以l只能等于2
            z = zs[-m]  # 第2层z2
            sp = sigmoid_prime(z)  # 第2层误差的前半部分，列向量形式，30*1
            delta = np.dot(self.weights[-m + 1].transpose(), delta) * sp  # 第二层误差，30*1 x 30*1 = 30*1(内积)
            nabla_b[-m] = delta  # 损失函数在第2层关于偏置的梯度，为第2层的误差，30*1
            nabla_w[-m] = np.dot(delta, activations[-m - 1].transpose())  # 关于权重的梯度,30*1 x 1*784 = 30*784

        # 返回梯度值
        return nabla_b, nabla_w

    def cost_derivative(self, output_activations, y):
        """
        传入参数：
            output_activations：最后一层（第3层）的输出向量，a3(_hat)
            y:一个样本的真实标签
            self.cost_derivative(activations[-1], y)

        传出参数：
            输出层误差的前半部分：（a3-y）=预估值-真实值
            列向量

        """
        return output_activations - y

    def evaluate(self, test_data):
        """
            输入：zip格式的测试数据
            过程：每个测试样本计算前向传播到a3(y_hat),取y_hat得最大值的索引为预估值
            输出：分类正确的个数
        """
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]  # argmax返回的是最大数的索引
        return sum(int(x == y) for (x, y) in test_results)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """
        输入：第三层的列向量z3,10行1列，numpy格式
              sigmoid_prime(zs[-1])

        输出：z3代入sigmoid函数，再求导。输出层误差的后半部分,10行1列，numpy格式
    """
    return sigmoid(z) * (1 - sigmoid(z))


# 构建输入层784个结点，隐藏层30个结点，输出层10个结点的三层前馈神经网络，使用训练数据集进行模型训练，使用测试数据集进行测试：
net = Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data)