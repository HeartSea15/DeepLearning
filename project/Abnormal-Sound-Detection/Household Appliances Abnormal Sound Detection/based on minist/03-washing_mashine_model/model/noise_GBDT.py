"""
    GBDT(Gradient Boosting Decision Tree): 梯度提升决策树

    介绍：
        1. 有监督的机器学习方法，解决分类问题和预测问题。
        2. GBDT融合了三方面的优势：提升Boosting、梯度Gradient和决策树Decision Tree
            提升Boosting：将多个弱分类器通过线性组合实现强分类器的过程；
            梯度Gradient：算法在Boosting过程中求解一般损失函数（非平方损失函数和指数损失函数）时增强了灵活性和便捷性.
                        关键是利用损失函数的负梯度在当前模型的值作为回归问题提升树算法中的残差的近似值，拟合一个回归树。
            决策树Decision Tree：算法所使用的弱分类器为CART决策树
        3. 因变量为离散值时：
            如果损失函数为指数损失函数，GBDT算法实际退化为AdaBoost算法；
            如果损失函数为对数似然损失函数，GBDT的残差类似于logistics回归的对数似然损失；

    优点：
        1. GBDT对数据类型不做任何限制，既可以是连续的数值型，又可以是离散的字符型（要做数值化或哑变量处理）
        2. 相对于SVM模型来说，较少参数的GBDT具有更高的准确率和更少的运算时间，面对异常数据时具有更强的稳定性。


    算法调用：
        分类问题：from sklearn import ensemble.GradientBoostingClassifier
        预测问题：from sklearn import ensemble.GradientBoostingRegressor

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
n_estimators = [100, 200, 400, 600]  # 基础分类器的数量
learning_rate = [0.01, 0.1, 0.2, 0.4]  # 学习率
min_samples_split = [2, 3, 4]     # 指定根节点或中间节点能够继续分割的最小样本量
min_samples_leaf = [1, 2, 3]      # 指定叶节点最小样本量

parameters = {'learning_rate': learning_rate,
              'n_estimators': n_estimators,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf,
              }

print(time.strftime('%Y-%m-%d %H:%M:%S'))
StartTime = time.time()
print('开始训练')
base_model = model_selection.GridSearchCV(GradientBoostingClassifier(),
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


save(filename='./parameters/GBDT.json')
print('model has been saved')
'''------------以上部分放gpu训练-----------------------------------'''
"""
运行时间太长，中间异常退出，故没有做参数调优，按照AdaBoost的参数训练测试的
"""




