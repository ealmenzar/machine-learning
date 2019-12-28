from sklearn import tree
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt
import numpy as np

"""

Decision Trees: Partition the dataset into a training and a testing set.
Run a decision tree learning algorithm usign the training set. Test the
decision tree on the testing dataset and report the total classification error
(i.e. 0/1 error). Repeat the experiment with a different partition. Plot
the resulting trees. Are they very similar, or very different? Explain why.
Advice: it can be convenient to set a maximum depth for the tree.

"""


def get_data(name):
    data = load_svmlight_file(name)
    return data[0], data[1]


def data_partition(N):
    x_train = xm[:N]
    y_train = y[:N]
    x_test = xm[N:]
    y_test = y[N:]
    return x_train, y_train, x_test, y_test


def classification_error(y_test, y_predict):
    tf = y_test == y_predict
    errors = np.where(tf == False)[0]
    return len(errors), len(tf)


x, y = get_data("australian.txt")
xm = x.toarray()  # type(x) = scipy.sparse.csr.csr_matrix, type(xm) = numpy.ndarray
N = 150  # size of training sets
d = 20  # max_depth

x_train, y_train, x_test, y_test = data_partition(N)

clf = tree.DecisionTreeClassifier(max_depth=d)
clf = clf.fit(x_train, y_train)

y_predict = clf.predict(x_test)

tree.plot_tree(clf)
plt.show()

errors, total = classification_error(y_test, y_predict)

print(errors, "/", total)
