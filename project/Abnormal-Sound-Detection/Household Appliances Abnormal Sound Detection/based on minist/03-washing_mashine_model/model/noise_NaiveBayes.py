"""
    NaiveBayes: 朴素贝叶斯
    介绍：
        1. 有监督的机器学习方法，专门解决分类问题的模型。
        2. 核心思想是：
                （1）通过已知类别的训练数据集，计算样本的先验概率，
                （2）然后利用贝叶斯公式测算未知类别样本属于某个类别的后验概率，
                （3）最终以最大后验概率所对应的类别作为样本的预测值。
    优点：
            运算简单高效；
            算法拥有古典概率的理论支撑，分类效率稳定；
            算法对缺失数据和异常数据不太敏感
    缺点：
            模型的判断结果依赖于先验概率，有一定的错误率；
            对输入的自变量X要求具有相同的特征（如变量均为数值型或离散型或0-1型）;
            模型的前提假设(自变量之间满足独立的假设条件)在实际应用中很难满足

    要点：
        几种数据类型下的贝叶斯模型, 用于计算P(X|Ci)：
            数据集中的自变量X均为连续的数值型(假设数值型变量的条件概率服从正太分布) -- 高斯贝叶斯分类器
            数据集的自变量X均为离散型变量(假设离散型变量的条件概率服从多项式分布) -- 多项式贝叶斯分类器
            数据集的自变量X均为0-1二元值(假设变量的条件概率服从伯努利分布) -- 伯努利贝叶斯分类器

        需要注意：以离散型变量为例，我们只是假设它是符合多项式分布，
                 如果它不是多项式分布，分类器的预测效果不会很理想，相反符合多项式分布，分类效果会不错

    算法调用：
        高斯贝叶斯分类器 -- naive_bayes.GaussianNB(priors=None)
        多项式贝叶斯分类器 -- naive_bayes.MultinomialNB(alpha=1.0, fit_prior = True, class_prior = None)
        伯努利贝叶斯分类器 -- naive_bayes.BernoulliNB(alpha=1.0, binarize=0.0, fit_prior = True, class_prior = None)

    z回复：
        只有数据量足够大时，prior统计才具准确性，
        同时数据才可能更符合高斯分布，否则数据的分布可能是长尾/多项式/伽马分布等各种奇怪形状。
        对于如何选择分布，对结果影响很大

"""

'''选择高斯贝叶斯分类器, 由于无需选择参数，在noise_load中直接训练和预测即可'''
