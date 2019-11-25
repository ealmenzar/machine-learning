import numpy as np
import random as rdm
from time import time

"""
    f. Modify the experiment such that xn ∈ R^10 instead of R^2. Run the algorithm
    on a randomly generated dataset of size 1000. 
    How many updates does the algorithm take to converge?
"""


def create_mset(N, M):
    xn = []
    for i in range(N):
        r = []
        for j in range(M):
            r0 = rdm.uniform(-1, 1)
            r.append(r0)
        xn.append(r)
    return xn


def map_mpoints(xn, n):
    y = []
    mappoints = []
    for s in xn:
        if np.dot(s, n) < 0:
            y.append(-1)
            mappoints.append([s, -1])
        else:
            y.append(1)
            mappoints.append([s, 1])
    return y, mappoints


def perceptron_malgorithm(N, w0, mapxn):
    mist = False
    end = False
    #print(mapxn)
    i = 0
    iterations = 0
    notfullcycle = False
    while not end:
        while not mist and i < N:
            if np.sign(np.dot(w0, mapxn[i][0])) != np.sign(mapxn[i][1]):
                mist = True
                notfullcycle = True
                mistpoint = mapxn[i]
            i += 1

        if mist:
            yx = [x * mistpoint[1] for x in mistpoint[0]]
            w0 = [w0a + w0b for w0a, w0b in zip(w0, yx)]
            mist = False
            iterations += 1
        elif notfullcycle:
            i = 0
            notfullcycle = False
            mist = False
        else:
            end = True
    return w0, iterations


def test_mperceptron(N, M):
    # We create the xn random set of M-dimensional points
    xn = create_mset(N, M)
    n = []
    for j in range(M):
        n0 = rdm.uniform(-1, 1)
        n.append(n0)
    y, mapxn = map_mpoints(xn, n)
    w = [0] * M
    start_time = time()
    g, it = perceptron_malgorithm(N, w, mapxn)
    elapsed_time = time() - start_time
    return g, it, elapsed_time


rdm.seed(30)
N = 1000
M = 10
g, it, el = test_mperceptron(N, M)
print(it)
print(el)
