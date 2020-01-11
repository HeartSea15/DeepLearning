"""
    提升方法Boosting：
        1. 提升方法是将弱学习方法提升为强学习算法的统计学习方法。
        2. 在分类学习中，提升方法根据反复修改训练数据的权值分布，构成一系列基本分类器，并将这些基本分类器线性组合，构成一个强分类器。
        3. 提升方法Boosting的两个问题：
            每一轮如何改变训练数据的权值或概率分布：提高那些被前一轮弱分类器错误分类样本的权值，使其在后一类的弱分类器收到更大关注
            如何将弱分类器组合成一个强分类器：加权多数表决。加大分类误差率小的弱分类器的权值，使其在表决中起到较大的作用
        4. 代表性的提升方法是AdaBoost

    AdaBoost：
        1. 是前向分布加法算法的特例，这是模型是有基本分类器组成的加法模型，损失函数是指数函数。
        2. AdaBoost是有监督的机器学习方法，解决分类问题和预测问题。
        3. AdaBoost是弱分类器的线性组合。

    AdaBoost算法步骤：
        1. 构建基础树f1(x)
        2. 计算基础树f1(x)的错误率e1
        3. 计算基础树f1(x)的权重a1
        4. 更新样本点的权重
        循环1,2,3,4

    提升树：
            以分类树或回归树为基本分类器的提升方法
            boosting + decision tree

    算法调用：
        分类问题（指数损失函数）：from sklearn import ensemble.AdaboostClassifier
        预测问题（平方误差损失函数）：from sklearn import ensemble.AdaboostRegressor
                 核心是利用第m轮基础树的残差值拟合第m+1轮基础树

        调参注意：
            在对Adaboost算法做交叉验证是，有两层参数需要调优，一个是基础模型的参数, 即DecisionTreeClassfier;
            另一个是提升树模型的参数，即AdaboostClassifier。在对基础模型调参时，参数字典的键必须以"base_estimator__"开头

"""
'''------------------------------模型导入-------------------------------------------'''
from function_import import *
swallowsound = read()
batch_size = 17800-50
X_train, y_train = swallowsound.train.next_batch(batch_size)
X_test = swallowsound.test.images[:4200]
y_test = swallowsound.test.labels[:4200]
'''------------------------------模型创建-------------------------------------------'''
max_depth = [3, 4, 5, 6]        # cart决策树的最大深度
n_estimators = [50, 100, 200, 300, 400, 500, 600]  # 基础分类器的数量
learning_rate = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6]  # 学习率
min_samples_split = [2, 3, 4]     # 指定根节点或中间节点能够继续分割的最小样本量
min_samples_leaf = [1, 2, 3]      # 指定叶节点最小样本量

parameters = {'learning_rate': learning_rate,
              'n_estimators': n_estimators,
              'base_estimator__max_depth': max_depth,
              'base_estimator__min_samples_split': min_samples_split,
              'base_estimator__min_samples_leaf': min_samples_leaf,
              }

print(time.strftime('%Y-%m-%d %H:%M:%S'))
StartTime = time.time()
print('开始训练')
base_model = model_selection.GridSearchCV(AdaBoostClassifier(base_estimator=DecisionTreeClassifier()),
                                          param_grid=parameters,
                                          scoring='roc_auc',
                                          cv=5,
                                          n_jobs=-1,
                                          verbose=1)
base_model.fit(X_train, y_train)
EndTime = time.time()
pass_time(EndTime - StartTime)
print('结束训练')

# 返回交叉验证后的最佳参数值
print('best_params', base_model.best_params_)
print('训练集的平均准确率best_score_', base_model.best_score_)

'''------------------------------模型保存-------------------------------------------'''
def save(filename):
    # if not os.path.exists(filename):
    #     os.mkdir('./parameters/')
    with open(filename, mode='w', encoding='utf-8') as f:
        data_save = {'model_best_param': list(base_model.best_params_.keys()),
                     'model_best_param_value': list(base_model.best_params_.values()),
                     'trainset_average_accuracy': base_model.best_score_.tolist(),
        }
        json.dump(data_save, f, ensure_ascii=False)  # ensure_ascii=False 确保中文写入不会出现乱码


save(filename='./parameters/AdaBoost.json')
print('model has been saved')
'''------------以上部分放gpu训练-----------------------------------'''
"""
为节省时间，其他参数固定，分别网格搜索的最大深度max_depth=3和learning_rate=0.2, 在此基础上再对其他参数调优

用时时间: hour:1  minute:57  second:45
best_params {'base_estimator__min_samples_leaf': 2,  'base_estimator__min_samples_split': 3, 'n_estimators': 600}
训练集的平均准确率best_score_ 0.9596800943263575

用时时间:hour:2  minute:26  second:50
best_params {'base_estimator__min_samples_leaf': 2, 'base_estimator__min_samples_split': 3, 'n_estimators': 3000}
训练集的平均准确率best_score_ 0.9683990475018512
"""



