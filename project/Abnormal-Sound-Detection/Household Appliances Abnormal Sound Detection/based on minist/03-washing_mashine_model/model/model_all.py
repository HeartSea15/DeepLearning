from function_import import *
swallowsound = read()
'''--------------------读取数据-----------------------------------'''
batch_size = 17800-50
X_train, y_train = swallowsound.train.next_batch(batch_size)
X_test = swallowsound.test.images[:4200]
y_test = swallowsound.test.labels[:4200]

'''-------------------模型构建-----------------------------'''
def svc_model():
    '''
        测试集上的准确率accuracy:  0.855
        查准率precision:  0.9046822742474916
        查全率recall:  0.7874818049490538
        AUC:  0.9289646156871791
    '''
    model = svm.SVC(kernel='rbf',
                    C=10,
                    probability=True,
                    gamma=1.6,
                    max_iter=1000)
    model.fit(X_train, y_train)
    return model


def knn_model():
    '''
        测试集上的准确率accuracy:  0.7409523809523809
        查准率precision:  0.8617100371747212
        查全率recall:  0.5623483745754488
        AUC:  0.7376959264181593
    '''
    model = neighbors.KNeighborsClassifier(n_neighbors=1,
                                           weights='uniform',
                                           leaf_size=10)
    model.fit(X_train, y_train)
    return model


def RandomForest_model():
    '''
        测试集上的准确率accuracy:  0.9095238095238095
        查准率precision:  0.9352667011910927
        查全率recall:  0.87627365356623
        AUC:  0.9660936345619431
    '''
    model = RandomForestClassifier(n_estimators=20,
                                   min_samples_split=2,
                                   min_samples_leaf=2,
                                   )
    model.fit(X_train, y_train)
    return model


def NaiveBayes_model():
    """
        测试集上的准确率accuracy: 0.6973809523809524
        查准率precision: 0.8464912280701754
        查全率recall: 0.4682193110140708
        AUC: 0.8063003135548564
    """
    model = naive_bayes.GaussianNB()
    model.fit(X_train, y_train)
    return model


def AdaBoost_model():
    """
    疑问：为什么给基分类器DecisionTreeClassifier赋值参数，运算速度超级慢，不赋值（默认参数）却很快？？？
        测试集上的准确率accuracy:  0.9014285714285715
        查准率precision:  0.9345646437994723
        查全率recall:  0.8592916060164969
        AUC:  0.9660483808587951

    """
    model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3,
                                                      min_samples_split=3,
                                                      min_samples_leaf=2),
                               learning_rate=0.2,
                               n_estimators=600,
                               )
    model.fit(X_train, y_train)
    return model


def GBDT_model():
    """
        损失函数默认为对数似然损失函数(loss='deviance')：
            测试集上的准确率accuracy:  0.9138095238095238
            查准率precision:  0.9609332609875203
            查全率recall:  0.8592916060164969
            AUC:  0.9695382466378996
        损失函数默认为对数似然损失函数(loss='exponential')：
            用时时间: hour:0  minute:15  second:45
            测试集上的准确率accuracy:  0.9157142857142857
            查准率precision:  0.9611021069692058
            查全率recall:  0.8631732168850073
            AUC:  0.9711539966505455

        与AdaBoost对比：
            比AdaBoost（0.9014）稍高0.037

    """
    model = GradientBoostingClassifier(max_depth=3,
                                       min_samples_split=3,
                                       min_samples_leaf=2,
                                       learning_rate=0.2,
                                       n_estimators=600,
                                       loss='exponential')
    model.fit(X_train, y_train)
    return model


def XGBoost_model():
    """
        用时时间: hour:0  minute:19  second:27
        测试集上的准确率accuracy:  0.92
        查准率precision:  0.9485179407176287
        查全率recall:  0.8850072780203785
        AUC:  0.9784374610835167
    """
    model = XGBClassifier(max_depth=10,
                          n_estimators=300,
                          learning_rate=0.2,
                          colsample_bytree=1,
                          gamma=0,
                          booster='gbtree')
    model.fit(X_train, y_train)
    return model


'''-------------------选择模型,参数修改-----------------------------'''
# svc_model_best_gamma(X_train, y_train)
StartTime = time.time()
model = XGBoost_model()       # 选择模型
EndTime = time.time()
pass_time(EndTime - StartTime)
print('结束训练')

filepath_confusion_matrix = './photo/XGBoost-confusion_matrix.jpg'
filepath_roc = './photo/XGBoost-roc.jpg'
filepath_ks = './photo/XGBoost-KS.jpg'

'''-------------------模型预测、评估、auc,ks,confusion matrix-----------------------------'''
model_predict_figure(model, X_test, y_test, filepath_confusion_matrix, filepath_roc, filepath_ks)

