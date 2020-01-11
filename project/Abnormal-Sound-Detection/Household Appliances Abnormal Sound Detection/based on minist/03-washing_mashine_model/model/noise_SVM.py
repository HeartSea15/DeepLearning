"""
    SVM: 某些支持向量构成的最大间隔的超平面
    介绍：
        1.有监督的机器学习方法，可用于离散因变量的分类和连续因变量的预测。
        2. 相对于其他单一的分类算法(logistics回归、决策树、朴素贝叶斯、KNN等)会有更好的预测准确率，
           主要是因为可以将低维线性不可分的空间转换为高维线性可分空间。
    优点：
        1. 鲁棒性好、避免维度灾难。svm模型最终形成的分类器仅依赖于一些支持向量，增减非支持向量的样本点不会影响。
        2. 很好的泛化能力
        3. 避免模型在运算过程中出现局部最优
    缺点：
        1. 不适合大样本的分类和预测，因为会消耗大量的计算资源和时间
        2. 模型对缺失样本非常敏感，需要建模前清洗好每一个观测样本（因为svm是基于距离度量的，所以对缺失样本敏感）
        3. 核函数的选择很敏感（根据经验，高斯核函数时往往拟合效果较好，因为高斯核是一种指数函数，它的泰勒展开可以是无穷维的，即相当于把原始样本点映射到高维空间去）
        4. 较回归和决策树，SVM为黑盒模型，对计算结果无法解释。

    关于SVM算法的调用：
        分类问题：
            线性可分SVM、近似线性可分SVM -- from sklearn.svm import LinearSVC
            非线性可分SVM -- from sklearn.svm import SVC
        预测问题：
            线性SVM回归 -- from sklearn.svm import LinearSVR
            非线性SVM回归 -- from sklearn.svm import SVR
"""
from function_import import *
swallowsound = read()
'''-------------------模型创建-----------------------------'''
batch_size = 17800-50
X_train, y_train = swallowsound.train.next_batch(batch_size)
X_test = swallowsound.test.images[:4200]
y_test = swallowsound.test.labels[:4200]

print(time.strftime('%Y-%m-%d %H:%M:%S'))
StartTime = time.time()

# 使用网格搜索法，选择非线性SVM“类”中的最佳C值
kernel = ['rbf', 'linear', 'poly', 'sigmoid']  # 分别为径向基核函数、线性核函数、多项式核函数、sigmoid核函数
C = [1, 10, 15]  # 惩罚系数，越大越有可能产生过拟合
gamma = [0.2, 0.6, 1.0, 1.6, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0]
max_iter = [1000]
probability = [True]
parameters = {'kernel': kernel,
              'gamma': gamma,
              'C': C,
              'max_iter': max_iter,
              'probability': probability}
print('开始训练')
grid_svc = model_selection.GridSearchCV(estimator=svm.SVC(),
                                        param_grid=parameters,
                                        scoring='accuracy',
                                        cv=10,
                                        n_jobs=20,
                                        verbose=1)
# 进行模型训练
grid_svc.fit(X_train, y_train)
EndTime = time.time()
pass_time(EndTime - StartTime)
print('结束训练')
# 返回交叉验证后的最佳参数值
print('best_params', grid_svc.best_params_)
print('训练集的平均准确率best_score_', grid_svc.best_score_)


def save(filename):
    # if not os.path.exists(filename):
    #     os.mkdir('./parameters/')
    with open(filename, mode='w', encoding='utf-8') as f:
        data_save = {'model_best_param': list(grid_svc.best_params_.keys()),
                     'model_best_param_value': list(grid_svc.best_params_.values()),
                     'trainset_average_accuracy': grid_svc.best_score_.tolist(),
        }
        json.dump(data_save, f, ensure_ascii=False)  # ensure_ascii=False 确保中文写入不会出现乱码


save(filename='./parameters/svm.json')
print('model has been saved')
'''------------以上部分放gpu训练-----------------------------------'''
'''
用时时间: hour:5  minute:38  second:19
best_params {'max_iter': 1000, 'probability': True, 'gamma': 1.6, 'C': 10, 'kernel': 'rbf'}
训练集的平均准确率best_score_ 0.8458028169014085

'''