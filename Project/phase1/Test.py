import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pickle


# X = pd.read_csv('data/X.csv', header=None, sep=" ", dtype=float)
# X = X.values
#
# y = pd.read_csv('data/y_bush_vs_others.csv', header=None)
# y_bush = y.values.ravel()
# y = pd.read_csv('data/y_williams_vs_others.csv', header=None)
# y_williams = y.values.ravel()
#
# knn = KNeighborsClassifier()
# result_knn = cross_validate(knn, X, y_bush, cv=(StratifiedKFold(n_splits=3, shuffle=True, random_state=1039)),
#                         scoring=('precision','recall','f1'), return_train_score=False)
#
# svc = SVC()
# result_svc = cross_validate(svc, X, y_bush, cv=(StratifiedKFold(n_splits=3, shuffle=True, random_state=1039)),
#                         scoring=('precision','recall','f1'), return_train_score=False)
mean_knn_b_n1 = 0.14894186825022562
mean_knn_b_n3 = 0.06264635195025688
mean_knn_b_n5 = 0.04573228856859605
best_svc_b = 0.6300969218929573

mean_knn_w_n1 = 0.16317016317016317
mean_knn_w_n3 = 0
mean_knn_w_n5 = 0
best_svc_w = 0.4794017094017095

pickle.dump((mean_knn_b_n1, mean_knn_b_n3, mean_knn_b_n5, best_svc_b),
            open('bush.pkl', 'wb'))
pickle.dump((mean_knn_w_n1, mean_knn_w_n3, mean_knn_w_n5, best_svc_w),
            open('williams.pkl', 'wb'))
