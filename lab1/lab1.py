from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from time import time


def get_data(name):
    data = load_svmlight_file(name)
    return data[0], data[1]


X, y = get_data("abalone.txt")
Xm = X.toarray()  # X type is scipy.sparse.csr.csr_matrix, Xm type is numpy.ndarray
Xm = Xm[:, np.newaxis, 2]  # We use only one feature
set_size = len(Xm)

"""
X, y = get_data("segment.scale.txt")
Xm = X.toarray()            # X type is scipy.sparse.csr.csr_matrix, Xm type is numpy.ndarray
Xm = Xm[:, np.newaxis, 1]   # We use only one feature
set_size = len(Xm)
"""


def plot_N(N):
    x_train = Xm[:N]
    y_train = y[:N]

    x_test = Xm[N:]
    y_test = y[N:]

    regr = linear_model.LinearRegression()
    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_test)

    plt.scatter(x_test, y_test, color='black')
    plt.plot(x_test, y_pred, color='blue', linewidth=1)
    plt.show()


def linear_regression(N):
    sizes = []
    errors = []
    segs = []

    while N < set_size:
        x_train = Xm[:N]
        y_train = y[:N]

        x_test = Xm[N:]
        y_test = y[N:]
        start_time = time()
        regr = linear_model.LinearRegression()
        regr.fit(x_train, y_train)
        y_pred = regr.predict(x_test)
        elapsed_time = time() - start_time
        segs.append(elapsed_time)
        error = mean_squared_error(y_test, y_pred)
        sizes.append(N)
        errors.append(error)
        N = N + 1

    plt.plot(sizes, errors, color='blue', linewidth=1)
    plt.show()
    plt.plot(sizes, segs, color='yellow', linewidth=1)
    plt.show()


def plot_log_N(N):
    x_train = Xm[:N]
    y_train = y[:N]

    x_test = Xm[N:]
    y_test = y[N:]

    regr = linear_model.LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    regr.fit(x_train, y_train)
    y_pred = regr.predict_proba(x_test)[:, 5] * 10

    plt.scatter(x_test, y_test, color='black')
    plt.scatter(x_test, y_pred, color='blue')

    plt.show()


def logistic_regression(N):
    sizes = []
    accur = []
    segs = []

    while N < set_size:
        x_train = Xm[:N]
        y_train = y[:N]
        x_test = Xm[N:]
        y_test = y[N:]

        start_time = time()
        regr = linear_model.LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
        regr.fit(x_train, y_train)
        acc = regr.score(x_test, y_test)
        elapsed_time = time() - start_time
        segs.append(elapsed_time)
        accur.append(acc)
        sizes.append(N)
        N = N + 10

    plt.plot(sizes, accur, color='blue', linewidth=1)
    plt.show()
    plt.plot(sizes, segs, color='yellow', linewidth=1)
    plt.show()
