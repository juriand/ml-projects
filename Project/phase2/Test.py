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
best_knn_b = 0.172720947253989
best_svc_b = 0.522980322429471

best_knn_w = 0.271111111111111
best_svc_w = 0.487482662491932

pickle.dump((best_knn_b, best_svc_b),
            open('bush.pkl', 'wb'))
pickle.dump((best_knn_w, best_svc_w),
            open('williams.pkl', 'wb'))
