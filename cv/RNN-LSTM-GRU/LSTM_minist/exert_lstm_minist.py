"""
    功能：循环神经网络LSTM完成分类问题
    数据集：minist
    在gpu上训练:
        单层静态LSTM网络--训练精度97.65%，测试精度98.43%
        单层静态GRU网络--训练精度97.65%，测试精度99.22%
"""

'''step1: 环境设定'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn


'''step2: 数据准备'''
mnist = input_data.read_data_sets('C:/tmp/data/mnist', one_hot=True)
# 查看一下数据维度
print(mnist.train.images.shape)
# 查看target维度
print(mnist.train.labels.shape)


'''step3: 超参数'''
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

n_input = 28  # MNIST数据输入, 一个时序具体的数据长度  即一共28个时序，一个时序送入28个数据进入LSTM网络
n_steps = 28  # 连续放28次，直到把784个像素点读完,表示时间序列总数
n_hidden = 128  # LSTM单元输出节点个数(即隐藏层个数)
n_classes = 10  # 总共类别数 (0-9数字)


'''step4: 准备好placeholder'''
x = tf.placeholder("float", [None, n_steps, n_input], name='X_placeholder')
y = tf.placeholder("float", [None, n_classes], name='Y_placeholder')


'''step5: 准备好权重/变量'''
weights = {'out': tf.Variable(tf.random_normal([n_hidden, n_classes]), name='Weight')}
biases = {'out': tf.Variable(tf.random_normal([n_classes]), name='bias')}


'''step6: 定义构建单层静态LSTM网络'''
def single_layer_static_lstm(x, weights, biases):
    """
    :param x: 输入张量, shape=[batch_size,n_steps,n_input]
    :param weights: shape=[n_hidden, n_classes]
    :param biases: shape=[n_classes]
    :return: 返回静态单层LSTM最后一个单元的输出
    """
    # 为了适应single_layer_static_lstm，把原始的输入调整成'n_steps'个(batch_size, n_input)的tensors
    x = tf.unstack(x, num=n_steps, axis=1)

    # 定义一个lstm cell
    lstm_cell = rnn.BasicLSTMCell(num_units=n_hidden, forget_bias=1.0)
    # gru_cell = tf.contrib.rnn.GRUCell(num_units=n_hidden)

    # 获取输出
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)  # 一共有n_steps(28)组输出outputs

    # 我们只取single_layer_static_lstm的最后一个输出
    return tf.matmul(outputs[-1], weights['out']) + biases['out']
pred = single_layer_static_lstm(x, weights, biases)


'''step7: 计算损失并指定optimizer'''
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y), name='loss')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

'''step8: 评估模型'''
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

'''step9: 初始化全部变量'''
init = tf.global_variables_initializer()

'''step10: 在session中执行graph定义的网络运算'''
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('./graphs/LSTM_MNIST', sess.graph)
    step = 1
    # 小于指定的总迭代轮数的情况下，一直训练
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # 用optimizer进行优化
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # 计算batch上的准确率
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # 计算batch上的loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")
    writer.close()

    # 测试集预测
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))