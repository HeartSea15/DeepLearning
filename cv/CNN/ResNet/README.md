# tensorflow 实现ResNet

## ResNet 介绍
ResNet(Residual Neural Network)由微软研究员的Kaiming He等4名华人提出，通过使用Residual Unit成功训练152层深的神经网络，在ILSVRC 2015的比赛中获得了冠军。取得了3.75%的top-5错误率，同时参数量却比VGGNet低。

在ResNet之前，瑞士教授Schmidhuber提出了Highway Network，原理与ResNet很相似。神经网络的深度对其性能非常重要，但是网络越深其训练难度越大，Highway Network的目标就是解决极深的神经网络难以训练的问题。Highway Network相当于修改了每一层的激活函数，此前的激活函数只是对输入做一个非线性变换 y=H(x,Wh),Highway NetWork 则允许保留一定比例的原始输入x，即 y=H(x,Wh)·T(x,Wt)+x·C(x, Wc)，其中T为变换系数，C为保留系数，论文中令C=1-T。
几百乃至上千层深的Highway Network可以直接使用梯度下降算法训练，并可以配合多种非线性激活函数，学习极深的神经网络变得可行了.

ResNet最初的灵感出自这个问题:在不断加深神经网络的深度时，会出现一个degradation的问题，即准确率会先上升然后达到饱和，再持续增加深度会导致准确率的下降。这并不是过拟合的问题，因为训练集的误差也会增大。假如有一个比较浅的网络达到了饱和的准确率，那么后面再加上几个y=x全等映射层，起码误差不会增加，即更深的网络不应该带来训练集上误差上升。而这里提到的使用全等连接将前一层输出传到后面的思想，就是ResNet的灵感来源。

在ResNet的第二篇论文中，提出了ResNet V2，区别是作者发现前馈和反馈信号可以直接传输，因此，非线性激活函数，直接替换为y=x。同时，在每一层都使用了BN。使新的残差单元将比以前更容易训练且泛化性更强.

## 参考博客
https://blog.csdn.net/m0_37917271/article/details/82346233
