# TensorFlow 实现Word2Vec

## 其他向量表示的方法介绍

（1）Bag of Words、One-Hot Encoder、Tf-Idf

缺点：丢失顺序信息

（2）2-gram、N-gram、共现矩阵

优点：考虑了词的顺序

缺点：词表膨胀

## Word2Vec介绍
One-Hot Encoder有一个问题，即我们对特征的编码往往是随机的，没有提供任何关联信息，没有考虑到字词间可能存在的关系

Word2Vec也称Word Embeddings，中文有很多叫法，比较普便的是“词向量”或“词嵌入”。Word2Vec是一个可以将语言中字词转为向量形式表达（Vector Representations）的模型

Word2Vec即是一种计算非常高效的，可以从原始语料中学习字词空间向量的预测模型。它主要分为CBOW（Continuous Bag of Words）和Skip-Gram两种模式，其中CBOW是从原始语句（比如：中国的首都是——）推测目标字词（比如：北京）；而Skip-Gram则正好相反，它是从目标字词推测出原始语句，其中CBOW对小型数据比较合适，而skip-gram在大型语料中表现得更好

## 本节任务
使用Skip-Gram模式的Word2Vec

