# Tensorflow实现Biderectional LSTM Classifier
## 双向递归神经网络(Bi-RNN)简介
普通的MLP对数据长度等有限制，而RNN虽然可以处理不固定长度的时序数据，但无法利用某个历史输入的未来信息。

双向递归神经网络（Bidirectional Recurrent Neural Networks， Bi-RNN）,是由Schuster和Paliwal于1997年首次提出的，和LSTM是在同一年被提出的。Bi-RNN的主要目标是增加RNN可利用的信息。RNN无法利用某个历史输入的未来信息，Bi-RNN则正好相反，它可以同时使用时序数据中某个输入的历史及未来数据.

## Bi-RNN应用
对Language modeling这类问题，可能Bi-RNN并不合适，因为我们的目标是通过前文预测下一个单词，这里不能将下文信息传给模型。对很多分类问题，比如手写文字识别、机器翻译等，使用Bi-RNN会大大提升模型效果。

## Bi-RNN结构简述
Bi-RNN网络结构的核心是把一个普通的单向的RNN拆分成两个方向，一个是随时序正向的，一个是逆着时序方向的。两个方向的RNN之间不会共用state，每一个时间节点的输入会分别传到正向和反向的RNN中，它们根据各自的状态产生输出，这两份输出会一起连接到Bi-RNN的输出节点，共同合成最终输出。

Bi-RNN中的每个RNN单元既可以是传统的RNN，也可以是LSTM单元或者GRU单元，同样也可以叠加多层Bi-RNN，进一步抽象的提炼出特征。如果最后使用作分类任务，我们可以将Bi-RNN的输出序列连接一个全连接层，或者连接全局平均池化Global Average Pooling，最后再接Softmax层，这部分和使用卷积神经网络部分一致

## Bi-RNN训练步骤
在使用BPTT（back-propagation through time）算法训练时，我们无法同时更新状态和输出。

同时，state在各自方向的开始处未知，这里需要人工设置。此外，state的导数在结尾处未知，这里一般需要设为0，代表此时对参数更新不重要。

训练步骤：

1. 对输入数据做forward pass操作，即inference的操作。我们先沿着1→T方向计算正向RNN的state，再沿着T←1方向计算反向RNN的state，然后获得output
2. backward pass操作，即对目标函数求导的操作。先对输出output求导，然后沿着T→1方向计算正向RNN的state的导数，再沿着1→T方向计算反向RNN的state导数
3. 根据求得的梯度值更新模型参数，完成一次训练。

## 本节任务
用TensorFlow实现一个Biderectional LSTM Classifier，数据集是minist。

## 总结
结果（accuracy）：

        训练集 -- 基本都是1
        
        测试集 -- 98.55%
        
Bidirectional LSTM Classifier 在mnist数据集上的表现虽然不如卷积神经网络，但也达到了一个不错的水平。 在图片这种空间结构显著的数据上不如卷积神经网络，但在无空间结构的单纯的时间序列上，相信Bi-LSTM会更具优势  
