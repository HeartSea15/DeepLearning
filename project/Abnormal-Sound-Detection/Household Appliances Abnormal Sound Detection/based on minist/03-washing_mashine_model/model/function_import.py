from swallowsound_input_data import read_data_sets
from sklearn.metrics import precision_score, recall_score, accuracy_score  # 查准率、查全率（正例覆盖率）、准确率
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve  # AUC值、roc曲线求fpr,tpr,
from sklearn.model_selection import cross_val_predict, cross_val_score  # ,交叉验证
from sklearn import metrics
from sklearn import model_selection

from sklearn import svm
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import naive_bayes
from xgboost import XGBClassifier

import time
import seaborn as sns
import math
import json
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']     # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False       # 用来正常显示负号


'''-------------------函数创建-----------------------------'''


def pass_time(time_):
    sum_time = math.floor(time_)

    h = math.floor(sum_time / 3600)
    m = math.floor((sum_time - h * 60 * 60) / 60)
    s = (sum_time - h * 60 * 60 - m * 60)

    print('\n用时时间:')
    print('hour:{0}  minute:{1}  second:{2}'.format(h, m, s))


def confusion_matrix(y_test, pred, filepath_confusion_matrix):
    cm = pd.crosstab(y_test, pred)  # 先实际再预测，对应：行是实际，列是预测
    cm = pd.DataFrame(cm)  # 将混淆矩阵构造成数据框，并加上字段名和行名称，用于行或列的含义说明
    print(cm)

    plt.figure(1)
    # 绘制热力图
    sns.heatmap(cm, annot=True, cmap='GnBu')
    # 添加x轴和y轴的标签
    plt.xlabel('Predict Lable')
    plt.ylabel('Real Lable')
    plt.title('confusion_matrix')
    plt.savefig(filepath_confusion_matrix)
    plt.close()


def plot_ks(y_test, y_score, filepath_ks, positive_flag):
    """
    自定义绘制ks曲线的函数,K-S曲线, KS一般大于0.4，模型可以接受
    :param y_test: 测试集实际值，series格式
    :param y_score: 测试集预测值，series格式
    :param positive_flag: 正例为1的标志
    :return:
    """
    # 对y_test,y_score重新设置索引
    y_test.index = np.arange(len(y_test))
    y_score.index = np.arange(len(y_score))

    # 构建目标数据集
    target_data = pd.DataFrame({'y_test': y_test, 'y_score': y_score})

    # 按y_score降序排列
    target_data.sort_values(by='y_score', ascending=False, inplace=True)

    # 自定义分位点
    cuts = np.arange(0.1, 1, 0.1)  # array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    # 计算各分位点对应的Score值
    index = len(target_data.y_score) * cuts
    scores = target_data.y_score.iloc[index.astype('int')]

    # 根据不同的Score值，计算Sensitivity真正例率 和 Specificity真负利率
    Sensitivity = []  # TPR
    Specificity = []  # FPR = 1-Specificity

    for score in scores:
        # 真正例率TPR = TP/(TP+FN) = 真正例/(真正例+假反例) = 真正例/(实际正例总数)
        # 正例覆盖样本数量与实际正例样本量
        positive_recall = target_data.loc[(target_data.y_test == positive_flag)
                                          & (target_data.y_score > score), :].shape[0]  # 正例覆盖样本数量
        positive = sum(target_data.y_test == positive_flag)  # 实际正例样本量

        # 假正例率FPR = FP/(TN+FP) = 假正例/(真反例+假正例) = 假正例/(实际反例总数)
        # 负例覆盖率specificity = 1 - FPR = 正确预测的负例数/实际负例数
        # 负例覆盖样本数量与实际负例样本量
        negative_recall = target_data.loc[(target_data.y_test != positive_flag)
                                          & (target_data.y_score <= score), :].shape[0]
        negative = sum(target_data.y_test != positive_flag)  # 实际负例样本量

        Sensitivity.append(positive_recall / positive)
        Specificity.append(negative_recall / negative)

    # 构建绘图数据
    plot_data = pd.DataFrame({'cuts': cuts, 'y1': 1 - np.array(Specificity), 'y2': np.array(Sensitivity),
                              'ks': np.array(Sensitivity) - (1 - np.array(Specificity))})
    print(plot_data)

    plt.figure()
    # 以分位数为x轴，以Sensitivity和1-Specificity分别为y轴，画两条线
    plt.plot([0] + cuts.tolist() + [1], [0] + plot_data.y1.tolist() + [1], label='FPR(1-Specificity)')
    plt.plot([0] + cuts.tolist() + [1], [0] + plot_data.y2.tolist() + [1], label='TPR(Sensitivity)')

    # 寻找Sensitivity和1-Specificity之差的最大值索引
    max_ks_index = np.argmax(plot_data.ks)
    # 添加参考线
    plt.vlines(plot_data.cuts[max_ks_index],  # 横坐标
               ymin=plot_data.y1[max_ks_index],
               ymax=plot_data.y2[max_ks_index],
               linestyles='--')
    #     # 添加文本信息
    plt.text(x=plot_data.cuts[max_ks_index] + 0.01,
             y=plot_data.y1[max_ks_index] + plot_data.ks[max_ks_index] / 2,
             s='KS= %.2f' % plot_data.ks[max_ks_index])
    plt.legend()
    plt.title('KS曲线')
    plt.savefig(filepath_ks)
    plt.close()


def plot_roc_curve(fpr, tpr, roc_auc, filepath_roc, label=None):
    plt.figure()
    # 绘制面积图
    plt.stackplot(fpr, tpr, color='steelblue', alpha=0.5, edgecolor='black')
    # 添加边际线
    plt.plot(fpr, tpr, color='black', lw=1, label=label)
    # 添加对角线
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    # 添加文本信息
    plt.text(0.5, 0.3, 'ROC curve (area = %0.2f)' % roc_auc)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('roc曲线')
    plt.savefig(filepath_roc)
    plt.close()


def svc_model_best_c(x_train, y_train):
    accuracy = []   # [0.6860365257400565,0.6543115475662715,0.6890809718963609]
    C = [1, 5, 10]
    for c in C:
        # 使用10重交叉验证的方法，对比每一个c值下svc模型的预测准确率
        result = cross_val_score(svm.SVC(kernel='rbf', C=c, probability=True, gamma=0.2, max_iter=1000),
                                 x_train,
                                 y_train,
                                 cv=10,
                                 scoring='accuracy',
                                 verbose=1
                                 )
        accuracy.append(result.mean())

    arg_max = np.array(accuracy).argmax()

    plt.plot(C, accuracy)
    plt.scatter(C, accuracy)
    plt.text(C[arg_max],accuracy[arg_max], '最佳c值为%s'% int(C[arg_max]))
    plt.xlabel('c')
    plt.ylabel('accuracy')
    plt.title('svc中最佳c值')
    plt.savefig('./photo/svc中最佳c值.jpg')
    plt.close()


def svc_model_best_gamma(X_train, y_train):
    accuracy=[0.6603340095776422, 0.8172397428590761, 0.8356617712000391, 0.8442809819958581, 0.8469853663149152,
     0.8472669614689876, 0.8470986135435974, 0.8358865194171416, 0.6598311130532853, 0.4956624476946896,
     0.4957744889203795] # 第二次运行结果
    # accuracy=[0.7159985563870236, 0.8232128424691583, 0.8388157163983934, 0.8477187758522453, 0.8477175063340011,
    #  0.8480001173744685, 0.8456335387422292, 0.8350974696253329, 0.6657648665743406, 0.49583025538281034,
    #  0.49611219987076655]  # 第一次运行结果
    # accuracy=[]
    gamma = [0.2, 0.6, 1.0, 1.2, 1.6, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0]
    # for i in gamma:
    #     # 使用10重交叉验证的方法，对比每一个c值下svc模型的预测准确率
    #     result = cross_val_score(svm.SVC(kernel='rbf', C=10, probability=True, gamma=i, max_iter=1000),
    #                              X_train,
    #                              y_train,
    #                              cv=10,
    #                              scoring='accuracy',
    #                              verbose=1
    #                              )
    #     print(result.mean())
    #     accuracy.append(result.mean())
    # print(accuracy)
    arg_max = np.array(accuracy).argmax()
    plt.plot(gamma, accuracy)
    plt.scatter(gamma, accuracy)
    plt.text(gamma[arg_max], accuracy[arg_max], '最佳gamma值为%s' % (gamma[arg_max]))
    plt.xlabel('gamma')
    plt.ylabel('train_avg_accuracy')
    plt.title('svc中最佳gamma值')
    plt.savefig('./photo/svc中最佳gamma值.jpg')
    plt.close()


def model_predict_figure(model, X_test, y_test, filepath_confusion_matrix, filepath_roc, filepath_ks):
    '''-------------------模型预测-----------------------------'''
    # 模型在测试集上的预测
    pred = model.predict(X_test)
    print('预测结果统计:\n', pd.Series(pred).value_counts())
    accuracy = metrics.accuracy_score(y_test, pred)
    print('测试集上的准确率accuracy: ', accuracy)
    print('查准率precision: ', precision_score(y_test, pred))
    print("查全率recall: ", recall_score(y_test, pred))
    '''-------------------模型评估-----------------------------'''
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_score)

    # 计算AUC的值, 评估标准0.8
    roc_auc = metrics.auc(fpr, tpr)
    print('AUC: ', roc_auc)
    # print(roc_auc_score(test_y, y_scores))  # 两种方法都可以

    print('模型的评估报告：\n', metrics.classification_report(y_test, pred))
    '''-----------------------画图--------------------------------------'''
    # 构建混淆矩阵
    confusion_matrix(y_test, pred, filepath_confusion_matrix)  # 先实际再预测，对应：行是实际，列是预测

    # roc曲线，AUC值
    plot_roc_curve(fpr, tpr, roc_auc, filepath_roc)

    # K-S
    plot_ks(pd.Series(y_test), pd.Series(y_score), filepath_ks, positive_flag=1)

    print('complete')


def load(filename):
    with open(filename, mode='r', encoding='utf-8') as f:
        data_load = json.load(f)  # json的load方法将字符串还原为我们的字典
    model_best_param = data_load['model_best_param']
    model_best_param_value = data_load['model_best_param_value']
    trainset_average_accuracy = data_load['trainset_average_accuracy']
    return model_best_param, model_best_param_value, trainset_average_accuracy


def read():
    num_classes = 2
    dir = './tmp/tensorflow/noise/input_2data250_fd/'
    swallowsound = read_data_sets(dir,
                                  gzip_compress=True,
                                  train_imgaes='train-images-idx3-ubyte.gz',
                                  train_labels='train-labels-idx1-ubyte.gz',
                                  test_imgaes='t10k-images-idx3-ubyte.gz',
                                  test_labels='t10k-labels-idx1-ubyte.gz',
                                  one_hot=False,
                                  validation_size=50,
                                  num_classes=num_classes,
                                  MSB=True)
    print(swallowsound.train.images.shape, swallowsound.train.labels.shape)  # 训练集
    print(swallowsound.test.images.shape, swallowsound.test.labels.shape)  # 测试集
    print(swallowsound.validation.images.shape, swallowsound.validation.labels.shape)  # 验证集
    return swallowsound
