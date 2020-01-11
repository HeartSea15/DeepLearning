"""
    决策树与随机森林：有监督的机器学习方法，可用于离散因变量的分类和连续因变量的预测
    概念：
        信息熵：某个事件所有可能值的熵和。
        条件熵：已知事件各取值下条件熵的期望。
        条件基尼指数：某变量各取值下条件基尼指数的期望
    要点：
        节点字段的选择：
            对离散型因变量进行分类：（叶节点的哪一类样本数量多就是哪一类）
                ID3算法： 信息增益（信息熵-条件熵），缺点是会偏向取值较多的字段 --多分支
                c4.5算法： 信息增益率（信息增益/信息熵），解决了信息增益的缺点 --多分支
                CART算法：条件基尼指数， --2分支
            对连续型因变量进行预测：（叶节点的样本均值作为该节点的预测值）
                CART算法：条件基尼指数， --2分支

        决策树的剪枝技术:
            预剪枝：限制树的层数、中间节点或叶节点的最小样本量、限制最多叶节点数量等
            后剪枝：
                误差降低剪枝法(自底向上) -- 结合测试集进行验证，可能导致剪枝过度
                悲观剪枝法(自顶向下)     -- 避免剪枝过程中使用测试集
                代价复杂度剪枝法        -- 平衡上升的误判率和下降的复杂度

        随机森林的实现思想：
            随机性体现在两方面：每棵树的训练样本、树中每个节点的分裂字段
            森林：多颗经过充分生长的cart决策树的集合
        随机森林的优点：
            避免单棵决策树过拟合、在计算量较低的情况下提高了预测准确率
    决策树的应用领域：
        医学上的病情诊断、金融领域的风险评估、销售领域的营销评估、工业产品的合格检验
    关于CART算法的调用：
        预测问题 -- DecisionTreeRegressor
        分类问题 -- DecisionTreeClassifier
    关于随机森林算法的调用：
        预测问题 -- RandomForestRegressor
        分类问题 -- RandomForestClassifier
"""
'''------------------------------模型导入-------------------------------------------'''
from function_import import *
swallowsound = read()
batch_size = 17800-50
X_train, y_train = swallowsound.train.next_batch(batch_size)
X_test = swallowsound.test.images[:4200]
y_test = swallowsound.test.labels[:4200]
'''------------------------------模型创建-------------------------------------------'''
n_estimators = [10, 15, 20]
min_samples_split = [2, 5]
min_samples_leaf = [1, 2]
parameters = {'n_estimators': n_estimators,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf}

print(time.strftime('%Y-%m-%d %H:%M:%S'))
StartTime = time.time()
print('开始训练')

grid_rf = model_selection.GridSearchCV(estimator=RandomForestClassifier(),
                                       param_grid=parameters,
                                       scoring='accuracy',
                                       cv=10,
                                       n_jobs=20,
                                       verbose=1)
grid_rf.fit(X_train, y_train)
EndTime = time.time()
pass_time(EndTime - StartTime)
print('结束训练')

# 返回交叉验证后的最佳参数值
print('best_params', grid_rf.best_params_)
print('训练集的平均准确率best_score_', grid_rf.best_score_)

'''------------------------------模型保存-------------------------------------------'''
def save(filename):
    # if not os.path.exists(filename):
    #     os.mkdir('./parameters/')
    with open(filename, mode='w', encoding='utf-8') as f:
        data_save = {'model_best_param': list(grid_rf.best_params_.keys()),
                     'model_best_param_value': list(grid_rf.best_params_.values()),
                     'trainset_average_accuracy': grid_rf.best_score_.tolist(),
        }
        json.dump(data_save, f, ensure_ascii=False)  # ensure_ascii=False 确保中文写入不会出现乱码


save(filename='./parameters/RandomForest.json')
print('model has been saved')
'''------------以上部分放gpu训练-----------------------------------'''

'''
用时时间:
hour:0  minute:16  second:44
结束训练
best_params {'min_samples_split': 2, 'min_samples_leaf': 2, 'n_estimators': 20}
训练集的平均准确率best_score_ 0.8867042253521127

'''


