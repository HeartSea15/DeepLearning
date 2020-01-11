import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']     # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False       # 用来正常显示负号

models = [('svm', 1),
          ('knn', 2),
          ('RandomForest', 3),
          ('NaiveBayes', 4),
          ('AdaBoost', 5),
          ('GBDT', 6),
          ('XGBoost', 7),
          ('deep_3conv', 8),
          ]

model_scores = pd.DataFrame({'Model': [name for name, _ in models],
                             'Accuracy': [0.855, 0.7409, 0.9095, 0.697, 0.9014, 0.9157, 0.92, 0.96119],
                             'precision': [0.905, 0.8617, 0.935, 0.846, 0.9345, 0.9611, 0.9485, 0.99],
                             'recall': [0.787, 0.5623, 0.876, 0.468, 0.85929, 0.8631, 0.885, 0.929],
                             # 'AUC': [0.93, 0.737, 0.966, 0.806, 0.966, 0.9711, 0.9784, ],
                             # 'K-S': [0.72, 0.48, 0.82, 0.49, 0.78, 0.81, 0.84, ]
                             })
model_scores.sort_values(by='Accuracy', ascending=False, inplace=True)
print(model_scores)


bar_width = 0.2
plt.bar(x=np.arange(len(models)), height=model_scores.Accuracy, label='Accuracy', color='red', width=bar_width)
plt.bar(x=np.arange(len(models))+bar_width, height=model_scores.precision, label='precision', color='blue', width=bar_width)
plt.bar(x=np.arange(len(models))+2*bar_width, height=model_scores.recall, label='recall', color='dimgrey', width=bar_width)

plt.xticks(np.arange(len(models))+0.2, np.array(model_scores['Model']))
plt.xlabel('Classifier')
plt.title('Comparison of accuracy, precision and recall rate between different classifiers')
plt.legend()
# plt.savefig('./photo/ac_pr_re.jpg')
plt.show()
