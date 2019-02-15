# Programming assignment 3

import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import math


num_set = 10
print("Loading datasets...")
Xs = pickle.load(open('binarized_xs.pkl', 'rb'))
ys = pickle.load(open('binarized_ys.pkl', 'rb'))
print("Done.")

l2_complexity = np.zeros((10, 15))
l1_num_zero_weight = np.zeros((10, 15))
l2_num_zero_weight = np.zeros((10, 15))
l2_train_cll = np.zeros((10, 15))
l2_test_cll = np.zeros((10, 15))
c = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
for i in range(num_set):
    # Split data into train set and test set
    X_train, X_test, y_train, y_test = train_test_split(Xs[i], ys[i], test_size=1.0/3, random_state=int("1039"))
    for d in range(15):
        # L2
        clf_l2 = LogisticRegression(penalty="l2", random_state=42, C=c[d])
        clf_l2.fit(X_train, y_train)
        W0 = clf_l2.intercept_[0]
        W = clf_l2.coef_[0]
        l2_complexity[i][d] += math.pow(W0, 2)
        if W0 == 0:
            l2_num_zero_weight[i][d] += 1
        for wi in W:
            l2_complexity[i][d] += math.pow(wi, 2)
            if wi == 0:
                l2_num_zero_weight[i][d] += 1

        for x in range(len(y_train)):
            l2_train_cll[i][d] += clf_l2.predict_log_proba(X_train)[x][y_train[x]*1]

        for x in range(len(y_test)):
            l2_test_cll[i][d] += clf_l2.predict_log_proba(X_test)[x][y_test[x]*1]

        # L1
        clf_l1 = LogisticRegression(penalty="l1", random_state=42, C=c[d])
        clf_l1.fit(X_train, y_train)
        W0 = clf_l1.intercept_[0]
        W = clf_l1.coef_[0]
        if W0 == 0:
            l1_num_zero_weight[i][d] += 1
        for wi in W:
            if wi == 0:
                l1_num_zero_weight[i][d] += 1

# Output result
print("Train set conditional log likelihood")
for i in range(num_set):
    print("\t".join("{0:.4f}".format(n) for n in l2_train_cll[i]))

print("\nTest set conditional log likelihood")
for i in range(num_set):
    print("\t".join("{0:.4f}".format(n) for n in l2_test_cll[i]))

for i in range(num_set):
    df_cmp = pd.DataFrame({'x': l2_complexity[i],
                           'y1': l2_train_cll[i], 'y2': l2_test_cll[i]})
    df_num = pd.DataFrame({'x': np.log10(c),
                           'y1': l1_num_zero_weight[i], 'y2': l2_num_zero_weight[i]})
    plt.title("Dataset " + str(i+1))
    plt.xlabel("Model complexity")
    plt.ylabel("Conditional log likelihood")
    plt.plot('x', 'y1', color='red', data=df_cmp, label='train_cll')
    plt.plot('x', 'y2', color='blue', data=df_cmp, label='test_cll')
    plt.legend()
    plt.show()

    plt.title("Dataset " + str(i+1))
    plt.xlabel("C")
    plt.ylabel("Number of zero weight")
    plt.plot('x', 'y1', color='red', data=df_num, label='l1')
    plt.plot('x', 'y2', color='blue', data=df_num, label='l2')
    plt.legend()
    plt.show()

# Generate a 'result.pkl' file
pickle.dump((l2_complexity, l2_train_cll, l2_test_cll, l2_num_zero_weight, l1_num_zero_weight), open('result.pkl', 'wb'))
