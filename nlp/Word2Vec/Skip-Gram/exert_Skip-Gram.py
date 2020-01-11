# 功能：用tensorflow完成一个Skip-Gram模式的Word2Vec的词向量生成。

'''Step 0:环境设定'''
import tensorflow as tf
import collections
import math
import os
import random
import zipfile
import numpy as np
import urllib

'''Step 1: Download the data'''
url = 'http://mattmahoney.net/dc/'
data_folder = 'data/'


def download(file_name, expected_bytes):
    """
    使用urllib.request.urlretrieve下载数据的压缩文件并核对文件尺寸，如果已经下载了文件则跳过
    :param file_name:
    :param expected_bytes: 核对文件的尺寸
    :return:
    """
    file_path = data_folder + file_name

    if os.path.exists(file_path):
        print("Dataset ready")
        return file_path

    # 使用urllib.request.urlretrieve下载数据的压缩文件
    file_name, _ = urllib.request.urlretrieve(url + file_name, file_path)

    # 获取文件相关属性
    file_stat = os.stat(file_path)

    # 比对文件的大小是否正确
    if file_stat.st_size == expected_bytes:  # .st_size,以字节为单位的大小
        print('Successfully downloaded the file', file_name)
    else:
        print(file_stat.st_size)
        raise Exception('File ' + file_name +
                        ' might be corrupted. You should try downloading it with a browser.')
    return file_path


file_path = download('text8.zip', 31344016)  # 'data/text8.zip'


'''Step 2: read the data'''
def read_data(file_path):
    with zipfile.ZipFile(file_path) as f:
        words = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return words


words = read_data(file_path)
print(words[:5])
print('Data size', len(words))


'''Step 3: Build the dictionary and replace rare words with UNK token'''
vocab_size = 50000
def build_vocab(words):
    """
    :param words: 词表
    :return: 转换后的编码（data）、
             每个单词的频数统计（count）、
             词汇表（dictionary）、
             及其反转的形式（reverse_dictionary）。
    """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocab_size - 1))

    # 生成 dictionary，词对应编号, word:id(0-4999),共50000个单词
    # 词频越高编号越小
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)

    # data把数据集的词都编号（以频数排序的编号）
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)

    # 记录UNK词的数量
    count[0][1] = unk_count

    # 编号对应词的字典：把字典的键和值对换位置
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


data, count, dictionary, reverse_dictionary = build_vocab(words)
del words
print('Most common words (+UNK)', count[:5])
print('数据集data, 以频数排序的编号', data[:5])
print('reverse_dictionary 编号对应词 ', [reverse_dictionary[i] for i in data[:5]])


'''Step 4: Function to generate a training batch for the skip-gram model.'''
data_index = 0


def generate_batch(batch_size, num_skips, skip_window):
    """
    :param batch_size: 一个batch的大小,即多少个样本
    :param num_skips: 对每个单词生成多少个样本
    :param skip_window: 单词最远可以联系的单词，如果为1，前后各一个单词。
    :return:
    """
    global data_index  # 全局索引，在data中的位置
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    # 建一个batch大小的数组，保存任意单词
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    # 建一个（batch，1）大小的二位数组，保存任意单词的前一个或者后一个单词，从而形成一个pair
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    # span为对某个单词创建相关样本时会使用到的单词数量
    span = 2 * skip_window + 1

    # 建一个结构为双向队列的缓冲区，大小不超过3
    buffer = collections.deque(maxlen=span)

    # 把span个单词顺序读入buffer作为初始值
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    # print(buffer)   # deque([5234, 3081, 12], maxlen=3)
    # 获取batch和labels
    for i in range(batch_size // num_skips):
        # 每次循环内对一个目标单词生成样本
        target = skip_window  # buffer中第skip_window个变量为目标单词
        targets_to_avoid = [skip_window]

        # 循环2次，一个目标单词对应两个上下文单词
        for j in range(num_skips):
            while target in targets_to_avoid:
                # 可能先拿到前面的单词也可能先拿到后面的单词
                target = random.randint(0, span - 1)  # 这里是从0,1,2中随机取一个，它包含两端的值
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

# 调用generate_batch函数简单测试一下其功能
# batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
# for i in range(8):
#     print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

'''Step 5: 超参数设定'''
batch_size = 128
embedding_size = 128   # 词向量的维度
skip_window = 1        # 单词间最远可以联系的距离
num_skips = 2          # 对每个目标单词提取的样本数

valid_size = 16        # 用来抽取的验证单词数
valid_window = 100     # 验证单词只从频数最高的100个单词中抽取
valid_examples = np.random.choice(valid_window, valid_size, replace=False)  # 生成验证数据valid_examples,False为不重复
num_sampled = 64       # 训练时用来做负样本的噪声单词的数量

'''Step 6: Build and train a skip-gram model'''
graph = tf.Graph()
with graph.as_default():

    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dateset = tf.constant(valid_examples, dtype=tf.int32)

    with tf.device('/cpu:0'):
        embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))  # 生成所有单词的词向量
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)  # 查找train_inputs对应的向量embed
        nce_weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size],
                                                      stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocab_size]))
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                             biases=nce_biases,
                                             labels=train_labels,
                                             inputs=embed,
                                             num_sampled=num_sampled,
                                             num_classes=vocab_size))
        # 定义优化器,学习速率为1.0
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        # 计算嵌入向量embeddings的L2范数nurm，L2范数：向量各元素的平方和再开根号
        # 计算验证单词的嵌入向量与词汇表中所有单词的相似性
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dateset)
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
        init = tf.global_variables_initializer()


'''Step 7: Begin training'''
num_steps = 1001  # 最大的迭代次数为10万次100001
with tf.Session(graph=graph) as session:
    init.run()
    print('Initialized')

    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        # 之后每2000次循环，计算一下平均loss并显示出来
        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            print('Average loss at step', step, ':', average_loss)
            average_loss = 0

        # 每10000次循环，计算一次验证单词与全部单词的相似度，并将与每个验证单词最相似的8个单词展示出来
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                # 根据id拿到对应单词
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8
                # 从大到小排序，排除自己本身，取前top_k个值
                nearest = (-sim[i, :]).argsort()[1:top_k+1]
                log_str = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)
    # 训练结束得到的词向量
    final_embeddings = normalized_embeddings.eval()
    # 观察前几组单词及其对应的词向量
    for i in range(5):
        print(reverse_dictionary[i], final_embeddings[i])

'''Step 8: Visualize the embeddings'''
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    """
    可视化Word2Vec效果的函数
    :param low_dim_embs: 降维到2维的单词的空间向量，将在图表中展示每个单词的位置
    :param labels:词频最高的100个单词
    :param filename: 图片命名
    :return:
    """
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),      # xy设置箭头尖的坐标
                     xytext=(5, 2),  # xytext设置注释内容显示的起始位置
                     textcoords='offset points',  # 设置注释文本的坐标系属性
                     ha='right',
                     va='bottom')

    plt.savefig(filename)


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)

plot_only = 100  # 展示词频最高的100个单词的可视化结果
# 每个词reverse_dictionary对应每个词向量final_embeddings
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
labels = [reverse_dictionary[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels)