"""
    XGBoost：DT→Boosting→BDT→GBDT→XGBoost
    介绍：
        1. 有监督的机器学习方法，可用于离散因变量的分类和连续因变量的预测。
        2. XGBoost的两点改进：
            (1). 损失函数加入了正则化项，用来控制模型复杂度，防止模型过拟合
            (2). GBDT模型在求解最优化问题是用了一阶导技术，而XGBoost则使用损失函数的一阶和二阶导。
     优点：
        (1).支持并行计算、提高算法的运行效率
        (2).损失函数加正则化项，用来控制模型复杂度，防止模型过拟合
        (3).传统的GBDT采用CART作为基分类器，XGBoost除了支持CART基础模型，还支持线性基础模型
        (4).传统的GBDT在每轮迭代时使用全部的数据，XGBoost则采用了与随机森林相似的策略，支持对数据进行采样

    算法调用：
        分类问题：from xgboost import XGBClassifier
        预测问题：from xgboost import XGBRegressor

"""
'''------------------------------模型导入-------------------------------------------'''
from function_import import *
swallowsound = read()
batch_size = 17800-50
X_train, y_train = swallowsound.train.next_batch(batch_size)
X_test = swallowsound.test.images[:4200]
y_test = swallowsound.test.labels[:4200]
'''------------------------------模型创建-------------------------------------------'''
parameters = {'max_depth': [3, 5, 10, 20],             # 决策树的最大深度,默认3层
              'n_estimators': [100, 200, 300],    # 基础分类器的数量，默认100个
              'learning_rate': [0.05, 0.1, 0.2, 0.5],  # 学习率,默认0.1
              'colsample_bytree': [0.8, 0.95, 1],      # 用于指定每个基础模型所需的采样字段比例，默认为1，表示使用原始数据的所有字段
              'gamma': [0, 1],                         # 用于指定节点分隔所需的最小损失函数下降值(惩罚项中叶子结点个数前的参数)，默认为0
              'booster': ['gbtree']}                   # 默认为'gbtree'，即cart模型，也可以是'gblinear',表示线性模型

print(time.strftime('%Y-%m-%d %H:%M:%S'))
StartTime = time.time()
print('开始训练')

base_model = model_selection.GridSearchCV(XGBClassifier(random_state=42),
                                          param_grid=parameters,
                                          scoring='accuracy',
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


save(filename='./parameters/XGBoost.json')
print('model has been saved')
'''------------以上部分放gpu训练-----------------------------------'''
"""
用时时间: hour:0  minute:53  second:45
best_params {'gamma': 0, 'n_estimators': 300, 'learning_rate': 0.2, 'booster': 'gbtree', 'colsample_bytree': 1, 'max_depth': 10}
训练集的平均准确率best_score_ 0.9092112676056338
"""