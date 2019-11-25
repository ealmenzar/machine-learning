import numpy as np
import random as rdm
import matplotlib.pyplot as plt
from time import time

"""

Generate a dataset of two-dimensional points, and choose a random line
in the plane as your target function f, where one side of the line maps
to +1 and the other side to −1.
Let the inputs xn ∈ R2 be random points in the plane, and evaluate the
target function f on each x_n to get the corresponding output y_n = f (x_n).
Experiment with the perceptron algorithm in the following settings:

    a. Generate a dataset of size 20. Plot the examples {(xn,yn)} as well as 
    the target function f on a plane.
    
    b. Run the perceptron algorithm on the dataset. Report the number of updates 
    that the algorithm takes before converging. Plot the examples {(xn, yn)}, the 
    target function f , and the final hypothesis g in the same figure.
    
    c. Repeat everything in b) with another randomly generated dataset of size 20, 
    and compare the result to b).
    
    d. Repeat everything in b) with another randomly generated dataset of size 100, 
    and compare the result to b).
    
    e. Repeat everything in b) with another randomly generated dataset of size 1000, 
    and compare the result to b).
    
"""


def create_set(N):
    xn = []
    for i in range(N):
        r0, r1 = rdm.uniform(-1, 1), rdm.uniform(-1, 1)
        xn.append([r0, r1])
    return xn


def map_points(xn, p0, p1):
    ex = []
    for s in xn:
        v0 = [p1[0] - p0[0], p1[1] - p0[1]]
        v1 = [p1[0] - s[0], p1[1] - s[1]]
        xp = np.cross(v0, v1)
        if xp > 0:
            ex.append([s, 1])
            plt.scatter(s[0], s[1], c='red')
        elif xp < 0:
            ex.append([s, -1])
            plt.scatter(s[0], s[1], c='blue')
        else:
            print("i is on the same line!")
            plt.scatter(s[0], s[1], c='yellow')
    return ex


def scatter_points(ex):
    for e in ex:
        if e[1] == 1:
            plt.scatter(e[0][0], e[0][1], c='red')
        elif e[1] == -1:
            plt.scatter(e[0][0], e[0][1], c='blue')
        else:
            plt.scatter(e[0][0], e[0][1], c='yellow')


def perceptron_algorithm(N, w0, target, ptarget, mapxn):
    mist = False
    end = False
    i = 0
    iterations = 0
    notfullcycle = False

    while not end:
        while not mist and i < N:
            if np.sign(np.dot(w0, mapxn[i][0])) != mapxn[i][1]:
                mist = True
                notfullcycle = True
                mistpoint = mapxn[i]
            i += 1

        if mist:
            yx = [x * mistpoint[1] for x in mistpoint[0]]
            w0 = [w0a + w0b for w0a, w0b in zip(w0, yx)]
            mist = False
            iterations += 1
            """
            nx = np.linspace(-1000, 1000, 100)
            ny = -w0[0] / w0[1] * target[0]
            plt.plot(nx, ny, c='green')
            plt.plot(target[0], target[1])
            scatter_points(mapxn)
            plt.ylim(-1300, 1300)
            plt.xlim(-1300, 1300)
            plt.show()
            """
        elif notfullcycle:
            i = 0
            notfullcycle = False
            mist = False
        else:
            end = True
    return w0, iterations


def test_perceptron(N):
    # We create the xn random set of two-dimensional points
    xn = create_set(N)

    # We create the random line that will determine wich points
    # are mapped as +1 and wich as -1 (target function)
    a = rdm.uniform(-3, 3)
    x = np.linspace(-1, 1, 10)
    y = a * x
    p0 = [-1, -a]   # Initial point of the line
    p1 = [1, a]     # Last point of the line
    plt.plot(x, y)

    # We map the points
    mapxn = map_points(xn, p0, p1)
    plt.ylim(-1.3, 1.3)
    plt.xlim(-1.3, 1.3)
    plt.show()

    w = [rdm.uniform(-0.1, 0.1), rdm.uniform(-0.1, 0.1)]
    target = [x, y]
    start_time = time()
    g, iterations = perceptron_algorithm(N, w, target, [p0, p1], mapxn)
    el = time() - start_time
    nx = np.linspace(-1, 1, 10)
    ny = -g[0] / g[1] * x
    plt.title("Final approximation")
    plt.plot(nx, ny, c='yellow')
    plt.plot(x, y)
    scatter_points(mapxn)
    plt.ylim(-1.3, 1.3)
    plt.xlim(-1.3, 1.3)
    plt.show()
    return g, iterations, el



rdm.seed(30)
N = 20
g, it, el = test_perceptron(N)
print(it)
print(el)

