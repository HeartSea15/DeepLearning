"""
    功能：用基于TensorFlow使用LSTM实现一个语言模型，语言模型是一个可以预测语句的概率模型
    日期：12/13/2019
"""


'''step1: 下载数据集PTB(Penn Tree Bank)和TensorFlowmodels库'''
'''PTB 文本数据集是语言模型学习中目前最广泛的数据集,
数据集中我们只需要利用 data 文件夹中的ptb.test.txt，ptb.train.txt，ptb.valid.txt 三个数据文件：测试，训练，验证 
这三个数据文件是已经经过预处理的，包含10000个不同的词语,有句尾的标记，同时将罕见的词汇统一处理为特殊字符'''
# 下载数据集PTB地址：http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
# 打开models/tutorials/rnn/pdb


'''step2: 载入工具库'''
# 导入常用的库，模型中的PTB reader主要是借助它读取数据内容，并把单词转为唯一的数字编码，以便神经网络处理
import numpy as np
import time
import tensorflow as tf
import reader


'''step3: 定义语言模型处理输入数据的class'''
class PTBInput(object):
    def __init__(self, config, data, name=None):
        """
        :param config: 配置参数
        :param data: data数据在训练集词表的频数由高到低的索引值
        :param name:
        """
        self.batch_size = batch_size = config.batch_size              # 一个batch的样本数
        self.num_steps = num_steps = config.num_steps                 # LSTM的展开步数，横向LSTM序列上有几个单元
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps  # 每个epoch内需要多少轮迭代

        # tensor格式，每次执行都会获取一个batch的数据
        self.input_data, self.targets = reader.ptb_producer(data, batch_size, num_steps, name=name)  # [batch_size, num_steps]


'''step4: 定义语言模型的class'''
class PTBModel(object):
    def __init__(self, is_training, config, input_):
        """
        :param is_training: 训练标记
        :param config: 配置参数
        :param input_: PTBInput类的实例input_
        """
        self._input = input_

        batch_size = input_.batch_size
        num_steps = input_.num_steps
        size = config.hidden_size       # LSTM的隐含节点数=词向量的维度
        vocab_size = config.vocab_size  # 词汇表的大小

        '''设置默认的LSTM单元'''
        def lstm_cell():
            """
            size: 隐含节点数，即前面的config.hidden_size == 词向量的维度
            forge_bias： 遗忘门的偏置
            state_is_tuple: True表示接受和返回的state将是2-tuple的形式
            """
            return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)

        attn_cell = lstm_cell  # 不加括号表示调用的是函数，加括号表示调用的是函数的结果

        if is_training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=config.keep_prob)
        # 堆叠前面的多层lstm_cell
        cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)
        # 设置LSTM单元的初始化状态为0
        self._initial_state = cell.zero_state(batch_size, tf.float32)

        '''创建网络的词嵌入embedding的部分'''
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, size], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        '''定义输出outputs'''
        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):  # 一般为了控制训练过程，限制梯度在反向传播时可以展开的步数为一个固定值num_steps
                if time_step > 0: tf.get_variable_scope().reuse_variables()  # 第二个循环开始设置复用变量,
                # input的三个维度分别为：batch中的第几个样本、样本中的第几个单词、单词向量的维度
                # input[:, time_step, :]表示所有样本的第time_step个单词
                (cell_output, state) = cell(inputs[:, time_step, :], state)  # cell_out, shape=[batch_size, hidden_size]
                outputs.append(cell_output)  # outputs, shape=(num_steps,batch_size,hidden_size)
        # concat后(batch_size, hidden_size * num_steps)
        # output为(batch_size * num_steps, hidden_size)
        output = tf.reshape(tf.concat(outputs, 1), [-1, size])

        '''损失函数计算'''
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
        logits = tf.matmul(output, softmax_w) + softmax_b  # shape=[batch_size * num_steps, vocab_size], 从隐藏语义转化成完全表示
        # loss, shape = [batch_size * num_steps]
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits],  # shape=[batch_size * num_steps, vocab_size]
                                                                  [tf.reshape(input_.targets, [-1])], # shape=[batch_size * num_steps]
                                                                  [tf.ones([batch_size * num_steps], dtype=tf.float32)]) # weight
        self._cost = cost = tf.reduce_sum(loss) / batch_size  # 平均到每个样本的误差
        self._final_state = state

        if not is_training:
            return

        '''定义学习速率的变量_lr、优化器等'''
        self._lr = tf.Variable(0.0, trainable=False)  # 定义学习速率，设为不可训练
        tvars = tf.trainable_variables()             # 获取全部可训练的参数
        # 针对前面得到的cost计算全部可训练的参数的梯度，用tf.clip_by_global_norm设置梯度的最大范数max_grad_norm，
        # 这即是gradient clipping的方法，控制梯度的最大范数，某种程度上起正则化的效果，可防止gradient explosion梯度爆炸的问题。
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        # 设置优化器
        optimizer = tf.train.GradientDescentOptimizer(self._lr)

        # 用optimizer.apply_gradients将前面clip过的梯度应用到所有可训练的参数tvars上，
        # 然后使用tf.contrib.framework.get_or_create_global_step()生成全局统一的训练步数
        self._train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                   global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")  # 用作控制学习速率
        self._lr_update = tf.assign(self._lr, self._new_lr)  # 将self._new_lr的值赋给当前的学习速率self._lr

    def assign_lr(self, session, lr_value):
        """
        外部控制模型的学习速率
        :param lr_value: 学习速率值
        """
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})
    '''整个模型的定义部分完成'''

    '''定义PTBModel class的一些property'''
    # @property装饰器可以将返回变量设为只读，防止修改变量引发的问题.
    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


'''step5: 定义几种不同大小的模型的参数'''
# 小模型的设置
class SmallConfig(object):
    init_scale = 0.1      # 网络中权重值的初始scale
    learning_rate = 1.0   # 学习速率的初始值
    max_grad_norm = 5     # 梯度的最大范数
    num_layers = 2        # LSTM可以堆叠的层数
    num_steps = 20        # LSTM反向传播的展开步数
    hidden_size = 200     # LSTM的隐含节点数
    max_epoch = 4         # 初始学习速率可训练的epoch数，在此之后需要调整学习速率
    max_max_epoch = 13    # 总共可训练的epoch数
    keep_prob = 1.0       # dropout层的保留节点的比例，设为1表示没有dropout
    lr_decay = 0.5        # 学习速率的衰减速度
    batch_size = 20       # 每个batch中样本的数量
    vocab_size = 10000    # 词表大小


# 中模型的设置
class MediumConfig(object):
    init_scale = 0.05     # 改小了，希望权重初值不要过大，小一些有利于温和的训练
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35        # 将LSTM反向传播的展开步数由20增大为35
    hidden_size = 650     # LSTM的隐含节点数增大约3倍
    max_epoch = 6         # 初始学习速率可训练的epoch数，4增大为6
    max_max_epoch = 39    # 总共可训练的epoch数增大约3倍
    keep_prob = 0.5       # dropout层的保留节点的比例
    lr_decay = 0.8        # 学习速率的衰减速度
    batch_size = 20
    vocab_size = 10000


# 大模型的设置
class LargeConfig(object):
    init_scale = 0.04     # 进一步缩小
    learning_rate = 1.0
    max_grad_norm = 10    # 5增大为10，大大放宽了最大梯度范数
    num_layers = 2
    num_steps = 35
    hidden_size = 1500    # 650上升到1500
    max_epoch = 14        # 相应增大
    max_max_epoch = 55    # 相应增大
    keep_prob = 0.35      # dropout层的保留节点的比例,因为模型复杂度的上升继续下降
    lr_decay = 1/1.15     # 学习速率的衰减速度进一步缩小
    batch_size = 20
    vocab_size = 10000


'''step6: 定义测试时的训练模型，参数都尽量使用最小值，只是为了测试可以完整运行模型'''
class TestConfig(object):
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


'''step7: 定义训练一个epoch数据的函数run_epoch'''
def run_epoch(session, model, eval_op=None, verbose=False):
    """
    :param session:
    :param model:
    :param eval_op: 评测操作
    :param verbose: 展示结果标记，训练时为True,其他False
    :return: perplexity, 平均cost的自然常数指数
    """
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)  # 执行model.initial_state初始化状态并获得初始状态

    # 创建输出结果的字典表
    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

            vals = session.run(fetches, feed_dict)
            cost = vals["cost"]
            state = vals["final_state"]

            costs += cost
            iters += model.input.num_steps

            if verbose and step % (model.input.epoch_size // 10) == 10:  # 每完成约10%的epoch，展示一次结果
                print("%.3f perplexity: %.3f speed : %.0f wps"
                      % (step * 1.0 / model.input.epoch_size,  # 当前epoch的进度
                         np.exp(costs / iters),  # perplexity, 平均cost的自然常数指数，越低代表模型输出的概率分布在预测样本上越好
                         iters * model.input.batch_size / (time.time() - start_time)))  # 训练速度（单词数每秒）

    return np.exp(costs / iters)


'''step8: 解压数据,配置config'''
raw_data = reader.ptb_raw_data('simple-examples/data/')
train_data, valid_data, test_data, _ = raw_data

config = SmallConfig()  # 训练模型的配置
eval_config = SmallConfig()  # 测试配置，并把batch_size和num_steps修改为1
eval_config.batch_size = 1
eval_config.num_steps = 1

'''step9: 创建graph'''
with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)  # 设置参数初始化
    with tf.name_scope("Train"):
        train_input = PTBInput(config=config, data=train_data, name="TrainInput")
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            m = PTBModel(is_training=True, config=config, input_=train_input)

    with tf.name_scope("Valid"):
        valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mvalid = PTBModel(is_training=False, config=config, input_=valid_input)

    with tf.name_scope("Test"):
        test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mtest = PTBModel(is_training=False, config=eval_config, input_=test_input)

    sv = tf.train.Supervisor()  # 创建训练的管理器sv
    with sv.managed_session() as session:
        for i in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)  # 计算累计的学习速率衰减值
            m.assign_lr(session, config.learning_rate * lr_decay)  # 初始学习速率×累计衰减值，并更新学习速率
            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))

            train_perplexity = run_epoch(session, m, eval_op=m.train_op, verbose=True)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

            valid_perplexity = run_epoch(session, mvalid)
            print("Epoch: %d valid Perplexity: %.3f" % (i + 1, valid_perplexity))

        test_perplexity = run_epoch(session, mtest)
        print("Test Perplexity: %.3f" % test_perplexity)