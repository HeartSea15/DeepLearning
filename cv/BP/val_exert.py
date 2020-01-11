"""
    作者：Heart Sea
    功能：np实现单隐层的BP算法，用到验证集, 正则化
    Model:
        1个输入层
        全连接
        1个隐含层, 激活ReLU,
        全连接
        1个输出层, 激活ReLU, 损失函数为交叉熵损失

    在3.0的基础上，改进如下：

        损失函数由二次代价函数改为交叉熵损失:避免了当出现“严重错误”时导致的学习缓慢
        L2正则化:改善过拟合

        权值初始化:权值初始化:设对l层有有n个输入神经元,使用均值为0,标准差为 1/sqrt(n)的高斯分布初始化l层的权值
                改善网络的学习速度

        早停止：每个epoch结束后（或每N个epoch后）： 在验证集上获取测试结果，
            随着epoch的增加，如果在验证集上发现测试误差上升，则停止训练；
            将停止之后的权重作为网络的最终参数

            因为精度都不再提高了，在继续训练也是无益的，只会提高训练的时间。
            那么该做法的一个重点便是怎样才认为验证集精度不再提高了呢?
            并不是说验证集精度一降下来便认为不再提高了，因为可能经过这个Epoch后，精度降低了，
            但是随后的Epoch又让精度又上去了，所以不能根据一两次的连续降低就判断不再提高。
            一般的做法是，在训练的过程中，记录到目前为止最好的验证集精度，
            当连续10次Epoch（或者更多次）没达到最佳精度时，则可以认为精度不再提高了。

    版本为3.0 的测试集准确率95.14%
    版本：4.0 的测试集准确率96.02%（还可以继续优化）
    日期：10/19/2019
    参考链接：http://neuralnetworksanddeeplearning.com/chap1.html
             https://www.jianshu.com/p/f9a14cec352d

    本代码执行顺序：  切分数据集，
                    训练（用到训练集和验证集），
                    保存模型和参数，
                    调用模型和参数，
                    测试集
"""

import gzip
import pickle
import random
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import json
import sys


# 加载MNIST数据集
def load_data():
    f = gzip.open('./data/mnist.pkl.gz')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return training_data, validation_data, test_data

# (50000, 784) (50000,)
# (10000, 784) (10000,)
# (10000, 784) (10000,)

# plt.imshow(image)
# plt.show()

# print(training_data[1][0])   # 训练集label的第一个数字5


#----------------构建支持向量机SVM模型，作为人工神经网络模型的性能对比的基模型-------
# 以svm模型为基模型（也可以用其他模型），神经网络不是号称学习能力强吗，准确率必须高于基模型。
def svm_baseline():
    training_data, validation_data, test_data = load_data()

    clf = svm.SVC()
    clf.fit(training_data[0], training_data[1])

    predictions = [int(a) for a in clf.predict(test_data[0])]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
    # zip后，两两对应一个元组，返回的是一个返回的是一个对象。如需展示列表，需手动 list() 转换，或者for
    # int(True)=1, int(False)=0,便于后面sum计算求和

    print("Baseline classifier using an SVM.")
    print(str(num_correct) + " of " + str(len(test_data[1])) + " values correct.")
#     10000条数据，9214条分类正确，正确率92.14%，如果用全部数据训练，94.35%

# svm_baseline()


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


#------------------------- 定义前馈神经网络模型 ------------------------------------

class QuadraticCost(object):
    # 损失函数为平方差损失
    @staticmethod
    def fn(a, y):
        # 平方损失函数
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        # 平方损失下的输出层误差
        return (a - y) * sigmoid_prime(z)


class CrossEntropyCost(object):
    """
        @staticmethod 静态方法只是名义上归属类管理，但是不能使用类变量和实例变量，是类的工具包
        放在函数前（该函数不传入self或者cls），所以不能访问类属性和实例属性
    """
    @staticmethod
    def fn(a, y):
        # 单个实例x的损失函数Cx为交叉熵损失函数
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    # numpy.nan_to_num(x):
    # 使用0代替数组x中的nan元素，使用有限的数字代替inf元素

    @staticmethod
    def delta(z, a, y):   # 改静态方法函数里不传入self 或 cls
        """
        :param z: zs[-1]
        :param a: activations[-1]
        :param y: y
        :return: 预测输出值 - 实际输出值 = 交叉熵损失函数下的输出层误差
        """
        return a - y


class Network2(object):

    def __init__(self, sizes, cost=CrossEntropyCost):  # Network2([784, 30, 10], cost=CrossEntropyCost)
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.cost = cost
        self.default_weight_initializer()

    def default_weight_initializer(self):
        """
        权值初始化:设对 𝑙 层有有 𝑛个输入神经元，使用均值为0，标准差为 1/sqrt(n)的高斯分布初始化 𝑙 层的权值。
        与正太分布初始化的相比优点：改善网络的学习速度
        """
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        # self.weights[0]第二层的初始化权重，(30, 784)
        # self.weights[1]为第三层的初始化权重，(10, 30)

        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        # biases[0]为第二层的初始化偏置  (30, 1)
        # biases[1]为第三层的初始化偏置   (10, 1)

    def feedforward(self, a):
        """
        :param a: 训练数据的单个实例（不包括label）
        :return: 前向传播求a3(预测输出y_hat)
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda=0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False,
            early_stopping_n=0):
        """
        :param training_data: 训练样本：包括实例和向量化的label，zip格式
        :param epochs: 30个循环
        :param mini_batch_size: 每组10个样本
        :param eta: 学习率 0.5
        :param lmbda: 正则化系数 5.0
        :param evaluation_data: validation_data
        :param monitor_evaluation_cost: True
        :param monitor_evaluation_accuracy: True
        :param monitor_training_cost: True
        :param monitor_training_accuracy: True
        :param early_stopping_n: 0表示不采用早停法，大于0表示采用早停法：定义验证集不更新多少次就停止训练
        :return: 定义批次随机梯度下降，
                返回evaluation_cost, evaluation_accuracy, training_cost, training_accuracy
        """

        training_data = list(training_data)  # zip打包为元组的列表,但是是一个对象，需要将zip格式列表化
        n = len(training_data)               # 5万个

        if evaluation_data:
            evaluation_data = list(evaluation_data)
            n_data = len(evaluation_data)

        # 采用早停法用到
        best_accuracy = 0
        # best_accuracy = 1
        no_accuracy_change = 0

        # 30个循环记录
        training_cost, training_accuracy = [], []
        evaluation_cost, evaluation_accuracy = [], []

        for j in range(epochs):  # 30次循环
            random.shuffle(training_data)   # 打乱训练集样本的顺序

            # n=5万个数据集,
            # mini_batch_size(10个样本作为一个一组)，一共有n/mini_batch_size=5万/10=5k组,每组10个样本
            mini_batches = [training_data[k: k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            # print(len(mini_batches))   # 5k组

            # 每一组mini_batch10个样本
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))

            print("-"*100)
            print("Epoch %s training complete" % j)

            # 训练集损失（正则化目标函数）
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)  # 一次循环
                training_cost.append(cost)  # 30次循环
                print("Cost on training data: {}".format(cost))

            # 训练集准确率
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)  # 一次循环
                training_accuracy.append(accuracy)   # 30次循环
                print("Accuracy on training data: {} / {}".format(accuracy, n))

            # 验证集损失（正则化目标函数）
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)  # 一次循环
                evaluation_cost.append(cost)  # 30次循环
                print("Cost on evaluation data: {}".format(cost))

            # 验证集准确率
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)  # 一次循环
                evaluation_accuracy.append(accuracy)   # 30次循环
                print("Accuracy on evaluation data: {} / {}".format(self.accuracy(evaluation_data), n_data))

            # 当验证集上的错误率不再下降时，就停止迭代
            if early_stopping_n > 0:
                """
                    采用早停法:
                """
                # 如果当前epoch在验证集上的accuracy大于前面的最好accuracy,记录
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    no_accuracy_change = 0
                    print("Early-stopping: Best so far {}".format(best_accuracy))

                # 否则记录no_accuracy_change的次数
                else:
                    no_accuracy_change += 1

                #  如果准确率不再上升累加的次数等于规定的次数，强制返回
                if (no_accuracy_change == early_stopping_n):
                    print("Early-stopping: No accuracy change in last epochs: {}".format(early_stopping_n))
                    return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):   # self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
        """
        :param mini_batch: 10个样本
        :param eta: 0.5的学习率
        :param lmbda: 正则化系数
        :param n: 训练样本数，5万
        :return: 每一组mini_batch更新当前的权重和偏置 ，即每10个样本更新一次b和w
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]   # 依次是第二层，第三层的偏置
        nabla_w = [np.zeros(w.shape) for w in self.weights]  # 依次是第二层，第三层的权重

        for x, y in mini_batch:

            # 对10个样本，取每个样本的x特征向量, y标签（one-hot）
            # x为784行，1列的向量，y为10行一列的标签
            # 依次对10个样本的两层，计算损失函数在各层的偏置梯度累加、权重梯度累加

            delta_nabla_b, delta_nabla_w = self.backprop(x, y)  # 损失函数对每一个样本各层的关于b、w的梯度
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]  # 10个样本，关于偏置梯度累加
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]  # 10个样本，关于权重梯度累加

        # 权重w考虑了正则化,注意这里用前面用 n ，而不是len(mini_batch)
        self.weights = [(1 - eta * (lmbda / n)) * w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]

        # 偏置b并不能增加模型复杂度 不需要正则化
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

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
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x      # 样本的784个维度，784*1
        activations = [x]   # 分层存储每层的输出项（z激活）,a向量, 依次是第一层a1（784*1）, 第二层a2（30*1）， 第三层a3（10*1）
        zs = []             # 分层存储每层的 z 向量,  依次是第二层z2（30*1），第三层z3（10*1）

        # 前向计算
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b    # 第2层维度：30*1，第3层维度：10*1, numpy类型
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        #  第3层误差：delta3（列向量与列向量的内积），10*1
        delta = self.cost.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta  # 损失函数在第三层关于偏置的梯度，等于第三层的误差，列向量, 10*1
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # 关于权重的梯度，10*30，损失函数在本层关于权值的梯度 = 本层误差 * 上一层的激活输出

        # 第2层误差：delta2，30*1
        for m in range(2, self.num_layers):
            z = zs[-m]  # 第2层z2
            sp = sigmoid_prime(z)   # 第2层误差的前半部分(当前层的z代入sigmoid函数再求导)，列向量形式，30*1
            delta = np.dot(self.weights[-m + 1].transpose(), delta) * sp  # 第二层误差，30*1 x 30*1 = 30*1(内积)
            nabla_b[-m] = delta     # 损失函数在第2层关于偏置的梯度，为第2层的误差，30*1
            nabla_w[-m] = np.dot(delta, activations[-m - 1].transpose())  # 关于权重的梯度,30*1 x 1*784 = 30*784

        # 返回梯度值
        return nabla_b, nabla_w

    # 计算准确率
    def accuracy(self, data, convert=False):
        if convert:
            # 针对训练集 self.accuracy(training_data, convert=True)
            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
        else:
            # 针对验证集或测试集 self.accuracy(evaluation_data)
            results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]

        result_accuracy = sum(int(x == y) for (x, y) in results)
        return result_accuracy

    # 计算训练集上的损失，这会加入L2规范化
    # 回到我们之前的L1规范化实现的问题，这里代码可改成
    # cost += (lmbda / len(data)) * sum(np.linalg.norm(w) for w in self.weights)
    def total_cost(self, data, lmbda, convert=False):
        """
        对于训练集，label已经是向量化，所以convert=False， self.total_cost(training_data, lmbda)
        对于验证集，label不是向量化， 所以convert=True，  self.total_cost(evaluation_data, lmbda, convert=True)
        label向量化是为了计算损失函数self.cost.fn
        :return: 全部数据的正则化目标函数
        """
        cost = 0.0

        for x, y in data:
            a = self.feedforward(x)        # 计算每个样本的a3
            if convert:
                y = vectorized_result(y)
            # 单个实例x的损失函数，所有训练数据的实例求和
            cost += self.cost.fn(a, y) / len(data)
            # 正则化目标函数
            cost += 0.5 * (lmbda / len(data)) * sum(np.linalg.norm(w) ** 2 for w in self.weights)

        return cost

    def evaluate(self, test_data):
        """
            输入：zip格式的测试数据
            过程：每个测试样本计算前向传播到a3(y_hat),取y_hat得最大值的索引为预估值
            输出：分类正确的个数
        """
        # print(type(test_data))
        # print(list(test_data))
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]  # argmax返回的是最大数的索引
        # print(list(test_data))
        return sum(int(x == y) for (x, y) in test_results)

    # 保存神经网络文件filename，保存网络的结构，权重，偏置，使用的损失函数。
    def save(self, filename):
        """
        size是一个列表，[784, 30, 10]
        w是numpy的array类型，调用它的tolist方法，把它转换成python的列表类型
        这里保存的是cost的类名字。(CrossEntropyCost)
        """
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


# 从filename加载神经网络，返回一个神经网络实例
def load(filename, test_data):
    """
    :param filename: 打开文件路径
    :param test_data: 测试数据集，zip格式
    :return: 测试集预测正确的个数
    这里用有个坑，zip对象” 是一个迭代器。 迭代器只能前进，不能后退。
    所以net.evaluate(test_data)之后运行list(test_data)为0
    """

    f = open(filename, "r")
    data = json.load(f)   # json的load方法将字符串还原为我们的字典
    f.close()

    # test_n = test_data
    # test_n = len(list(test_n))

    # cost = getattr(sys.modules[__name__], data["cost"])
    cost = data["cost"]
    net = Network2(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    test_data_accuracy = net.evaluate(test_data)

    print("测试集 {} / {}".format(test_data_accuracy, 10000))
    return net


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """
        输入：第三层的列向量z3,10行1列，numpy格式
              sigmoid_prime(zs[-1])

        输出：z3代入sigmoid函数，再求导。输出层误差的后半部分,10行1列，numpy格式
    """
    return sigmoid(z) * (1 - sigmoid(z))


if __name__ == '__main__':

#------------------------- 显示数据集维度 ------------------------------------
  # training_data, validation_data, test_data = load_data()
  # image = training_data[0][0].reshape(28, 28)  # 训练集label的第一个图片，每个样本（图片）28行，28列，即784维特征
  #
  # print(training_data[0].shape, training_data[1].shape)
  # print(validation_data[0].shape, validation_data[1].shape)
  # print(test_data[0].shape, test_data[1].shape)


#------------------------- 封装数据集成zip格式 ------------------------------------
  training_data, validation_data, test_data = load_data_wrapper()  # 三个数据集都是zip格式


#----------------------------- 训练并保存模型 -------------------------------------

  # 构建输入层784个结点，隐藏层30个结点，输出层10个结点的三层前馈神经网络，使用训练数据集进行模型训练，使用测试数据集进行测试：
  net = Network2([784, 30, 10], cost=CrossEntropyCost)
  net.SGD(training_data, 1, 10, 0.5, lmbda=5.0, evaluation_data=validation_data,
         monitor_evaluation_cost=True,
         monitor_evaluation_accuracy=True,
         monitor_training_cost=True,
         monitor_training_accuracy=True)

  net.save('./data/save_model')

#----------------------------- 调用模型后验证测试集 -------------------------------------
  load('./data/save_model', test_data)