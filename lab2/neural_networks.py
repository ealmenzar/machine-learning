from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import validation_curve
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt
import numpy as np

"""

Neural Networks: Train a Multi-Layer perceptron using the cross-entropy loss with l2 regularization 
(weight decay penalty). In other words, the activation function equals the logistic function. Plot curves 
of the training and validation error as a function of the penalty strength alpha. How do the curves behave? 
Explain why.

Advice: use a logarithmic range for hyper-parameter alpha. Experiment with different sizes of the training/validation 
sets and different model parameters (network layers).

"""


def get_data(name):
    data = load_svmlight_file(name)
    return data[0], data[1]


x, y = get_data("australian.txt")
xm = x.toarray()        # type(x) = scipy.sparse.csr.csr_matrix, type(xm) = numpy.ndarray
N = 100                 # size of training sets


def data_partition(N):
    x_train = xm[:N]
    y_train = y[:N]
    x_test = xm[N:]
    y_test = y[N:]
    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = data_partition(N)

clf = MLPClassifier(activation='logistic', alpha=1e-5, random_state=1)
clf.fit(x_train, y_train)

param_range = np.logspace(-7, 3, 3)
train_scores, test_scores = validation_curve(MLPClassifier(activation='logistic', alpha=1e-5, random_state=1),
                                              x_train, y_train, "alpha", param_range, cv=5)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with MLP")
plt.xlabel(r"$\alpha$")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()
