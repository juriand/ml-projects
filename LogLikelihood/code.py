# Programming assignment 2

import pandas as pd
import pickle
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math

print("Loading datasets...")
Xs = pickle.load(open('binarized_xs.pkl', 'rb'))
ys = pickle.load(open('binarized_ys.pkl', 'rb'))
print("Done.")

train_jll = np.zeros((10, 15))
test_jll = np.zeros((10, 15))
a = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
for i in range(10):
    # Split data into train set and test set
    X_train, X_test, y_train, y_test = train_test_split(Xs[i], ys[i], test_size=1.0/3, random_state=int("1039"))
    for d in range(15):
        clf = BernoulliNB(alpha=a[d])
        clf.fit(X_train, y_train)

        # Train set joint log likelihood
        jll = 0
        joint_log = clf._joint_log_likelihood(X_train)
        for n in range(len(X_train)):
            iy = y_train[n]*1
            jll = jll + joint_log[n][iy]
        train_jll[i][d] = jll

        # Test set joint log likelihood
        jll = 0
        clf.predict(X_test)
        joint_log = clf._joint_log_likelihood(X_test)
        for n in range(len(X_test)):
            iy = y_test[n]*1
            jll = jll + joint_log[n][iy]
        test_jll[i][d] = jll

# Output result
print("Train set joint log likelihood")
for i in range(10):
    print("\t".join("{0:.4f}".format(n) for n in train_jll[i]))

print("\nTest set joint log likelihood")
for i in range(10):
    print("\t".join("{0:.4f}".format(n) for n in test_jll[i]))

# for i in range(10):
#     plt.plot(a, train_jll[i])
#     plt.show()
#     plt.plot(a, test_jll[i])
#     plt.show()
#     print()

# Generate a 'results.pkl' fil
pickle.dump((train_jll, test_jll), open('results.pkl', 'wb'))