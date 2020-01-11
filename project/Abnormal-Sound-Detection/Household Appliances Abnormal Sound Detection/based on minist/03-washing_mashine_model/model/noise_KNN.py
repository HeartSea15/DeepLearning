"""
    KNN: K近邻算法，搜寻最近的k个已知类别样本用于未知类别样本的预测
    介绍：
        1. 有监督的机器学习方法，可用于离散因变量的分类和连续因变量的预测。
        2. 核心思想是比较已知y值的样本与未知y值样本的相似度，然后寻找最相似的k个样本用作未知样本的预测
        3. 对应离散因变量来说，从k个最近的已知类别样本中挑选出频率最高的类别用于未知样本的判断；
        4. 对于连续因变量来说，从k个最近的已知类别样本均值用于未知样本的预测
    缺点：
        （1）计算复杂度高、空间复杂度高，对大量测试样本计算量大，资源开销大。
        （2）K值的选择：最大的缺点是当样本不平衡时，如一个类的样本容量很大，而其他类样本容量很小时，有可能导致当输入一个新样本时，该样本的K个邻居中大容量类的样本占多数。
        （3） KNN是一种消极学习方法、懒惰算法。 缺少训练阶段，无法应对多样本
    要点：
        最佳k值的选择：
            设置近邻样本的投票权重，可以将权重设置为距离的倒数；
            多重交叉验证
        相似度的度量方法：
            欧氏距离--两点之间的直线距离
            曼哈顿距离--两点在轴上的相对距离总和
        近邻样本的搜寻方法：
            暴力搜寻法 -- 适合小样本的数据集，相当于两层for循环
            KD搜寻法 -- K指训练集包含的变量个数。根据经验，当数据集的变量个数超过20时，KD树的运行效率会被拉低。
            球树搜寻法 -- 避免KD搜寻法导致的'角'对搜寻速度的影响。


    关于KNN算法的调用：
        分类问题 -- from sklearn.neighbors import KNeighborsClassifier
        预测问题 -- from sklearn.neighbors import KNeighborsRegressor

        重要参数：n_neighbors, weights，实际应用项目中要对比各种可能的值。
"""
'''------------------------------模型导入-------------------------------------------'''
from function_import import *
swallowsound = read()
batch_size = 17800-50
X_train, y_train = swallowsound.train.next_batch(batch_size)
X_test = swallowsound.test.images[:4200]
y_test = swallowsound.test.labels[:4200]
'''------------------------------模型创建-------------------------------------------'''
K = np.arange(1, np.ceil(np.log2(len(X_train))))
n_neighbors = [int(i) for i in K]
leaf_size = [10, 30, 50, 80]
weights = ['uniform', 'distance']
parameters = {'n_neighbors': n_neighbors,
              'weights': weights,
              'leaf_size': leaf_size}

print(time.strftime('%Y-%m-%d %H:%M:%S'))
StartTime = time.time()
print('开始训练')

grid_knn = model_selection.GridSearchCV(estimator=neighbors.KNeighborsClassifier(),
                                        param_grid=parameters,
                                        scoring='accuracy',
                                        cv=10,
                                        n_jobs=20,
                                        verbose=1)
grid_knn.fit(X_train, y_train)
EndTime = time.time()
pass_time(EndTime - StartTime)
print('结束训练')

# 返回交叉验证后的最佳参数值
print('best_params', grid_knn.best_params_)
print('训练集的平均准确率best_score_', grid_knn.best_score_)

'''------------------------------模型保存-------------------------------------------'''
def save(filename):
    # if not os.path.exists(filename):
    #     os.mkdir('./parameters/')
    with open(filename, mode='w', encoding='utf-8') as f:
        data_save = {'model_best_param': list(grid_knn.best_params_.keys()),
                     'model_best_param_value': list(grid_knn.best_params_.values()),
                     'trainset_average_accuracy': grid_knn.best_score_.tolist(),
        }
        json.dump(data_save, f, ensure_ascii=False)  # ensure_ascii=False 确保中文写入不会出现乱码


save(filename='./parameters/knn.json')
print('model has been saved')
'''------------以上部分放gpu训练-----------------------------------'''
# best_params {'n_neighbors': 1, 'weights': 'uniform', 'leaf_size': 10}
# 用时时间:
# hour: 11 minute: 37 second: 11
# 训练集的平均准确率best_score_ 0.7329014084507042



