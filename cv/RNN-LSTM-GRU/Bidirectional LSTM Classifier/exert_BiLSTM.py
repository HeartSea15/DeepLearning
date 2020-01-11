"""
    功能：TensorFlow实现一个Biderectional LSTM Classifier
    数据集：minist
    日期：12/18/2019
    参考：TensorFlow实战

    结果（accuracy）：
        训练集 -- 基本都是1
        测试集 -- 98.55%
"""

'''step1: 导入模块'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("c:/tmp/data/mnist", one_hot=True)

'''step2: 设置训练参数'''
learning_rate = 0.01   # 学习速率，由于使用Adam优化器，所以学习速率较低
max_samples = 400000   # 最大训练样本数(迭代次数 × batch_size)
batch_size = 128
display_step = 10      # 每间隔10次训练就展示一次训练情况

'''step3: 设置网络参数'''
n_input = 28  # 因为minist图像是28×28，因此输入为28（图像的宽）
n_steps = 28  # LSTM的展开步数（unrolled steps of LSTM）,也设置为28（图像的高），这样图像的全部信息就用上了
# 这里是一次读取一行像素（28个像素点），然后下一个时间点再传入下一行像素点。

n_hidden = 256   # LSTM隐藏节点数
n_class = 10     # 分类数目

'''step4: 创建输入x和学习目标y的placeholder'''
# 样本被理解为一个时间序列，第一个维度是时间点n_steps, 第二个维度是每个时间点的数据n_input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_class])

'''step5: 创建最后的softmax层的权重、偏置'''
weights = tf.Variable(tf.random_normal([2*n_hidden, n_class]))  # 因为是双向LSTM,所以weights参数量翻倍
biases = tf.Variable(tf.random_normal([n_class]))

'''step6: 定义Biderectional LSTM网络的生成函数'''
def BiRNN(x, weights, biases):
    """
    :param x: 输入，[batch_size, n_steps, n_input]
    :param weights: softmax层的权重, [2*n_hidden, n_class]
    :param biases:  softmax层的偏置， [n_class]
    :return: 对双向LSTM的输出结果outputs做一个矩阵乘法加偏置
    """
    # 数据处理成LSTM单元的输入形式
    x = tf.transpose(x, [1, 0, 2])    # 前两个维度交换为[n_steps, batch_size, n_input]
    x = tf.reshape(x, [-1, n_input])  # 变形为（n_steps*batch_size, n_input）的形状
    x = tf.split(x, n_steps)          # n_steps个[batch_size, n_input]的列表
    # 以上三行等价于 x = tf.unstack(x, num=n_steps, axis=1)

    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    outputs, _ , _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell,
                                                             lstm_bw_cell,
                                                             x,
                                                             dtype=tf.float32)
    return tf.matmul(outputs[-1], weights) + biases
pred = BiRNN(x, weights, biases)

'''step7: 定义loss，优化器'''
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

'''step8: 评估模型'''
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accurary = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

'''step9: 初始化全部变量'''
init = tf.global_variables_initializer()

'''step10: 训练和测试'''
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < max_samples:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        if step % display_step == 0:
            acc = sess.run(accurary, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter" + str(step*batch_size) + ", Minibatch Loss=" + \
                  "{:.6f}".format(loss) + "Training Accuracy=" + \
                  "{:.5f}".format(acc))
        step += 1

    # 全部训练完后，对测试集预测
    test_len = 10000
    test_data = mnist.test.images[: test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", sess.run(accurary, feed_dict={x: test_data, y: test_label}))