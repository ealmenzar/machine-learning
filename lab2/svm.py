from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt
import numpy as np
import random

"""

Support Vector Machines: Run SVM to train a classifier, using radial basis as kernel function. 
Apply cross-validation to evaluate different combinations of values of the model hyper-parameters 
(box constraint C and kernel parameter γ). How sensitive is the cross-validation error to changes 
in C and γ? Choose the combination of C and γ that minimizes the cross-validation error, train the 
SVM on the entire dataset and report the total classification error.
Advice: use a logarithmic range for γ.

"""


def get_data(name):
    data = load_svmlight_file(name)
    return data[0], data[1]


x, y = get_data("australian.txt")
xm = x.toarray()        # type(x) = scipy.sparse.csr.csr_matrix, type(xm) = numpy.ndarray
N = 300                 # size of training sets


def data_partition(N):
    x_train = xm[:N]
    y_train = y[:N]
    x_test = xm[N:]
    y_test = y[N:]
    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = data_partition(N)

error = []
sc = []
E = []
G = []
C = []
A = []

for i in range(15):
    G.append(random.uniform(1.0, 1.005))
G.sort()

C.append(1.0)
for i in range(14):
    C.append(random.uniform(1.0, 10.0))
C.sort()

for i in C:
    error.clear()
    sc.clear()
    for j in G:
        clf = svm.SVC(C=i, gamma=np.log(j), kernel='rbf')
        clf.fit(x_train, y_train)
        scores = cross_val_score(clf, x, y, cv=5)
        sc.append(scores.mean())
        error.append((1 - scores).mean())

    E.append(error)
    A.append(sc)

# Plot for inaccuracy

fig, ax = plt.subplots()
im = ax.imshow(E, cmap="plasma")

ax.set_xlabel("gamma")
ax.set_ylabel("C")
ax.set_xticks(np.arange(len(G)))
ax.set_yticks(np.arange(len(C)))

roundG = []
for g in G:
    roundG.append(round(np.log(g), 4))

roundC = []
for c in C:
    roundC.append(round(c, 4))

ax.set_xticklabels(roundG)
ax.set_yticklabels(roundC)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

plt.title("Inaccuracy")
plt.colorbar(im)

plt.show()

# Plot for accuracy

fig2, ax2 = plt.subplots()
im = ax2.imshow(A, cmap="plasma")

ax2.set_xlabel("gamma")
ax2.set_ylabel("C")
ax2.set_xticks(np.arange(len(G)))
ax2.set_yticks(np.arange(len(C)))

roundG = []
for g in G:
    roundG.append(round(np.log(g), 4))

roundC = []
for c in C:
    roundC.append(round(c, 4))

ax2.set_xticklabels(roundG)
ax2.set_yticklabels(roundC)

plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

plt.title("Accuracy")
plt.colorbar(im)
plt.show()
