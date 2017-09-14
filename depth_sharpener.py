import cv2
import sys
import numpy as np
import numpy.random
import scipy.sparse as sp
import scipy.sparse.linalg
import PIL.Image
from io import BytesIO
import IPython.display
import numpy as np

def showarray(a, fmt='png'):
    pass


def abs_comp_matrix(img):
    return sp.diags([1.0/(np.abs(img.flatten()) + 1e-5)], [0])


def relu_comp_matrix(img, V):
    return sp.diags([1.0/(np.maximum(np.abs(img.flatten()), V) - V + 1e-5)], [0])


def soft_relu_comp_matrix(img, V):
    soft = np.log(1 + np.exp(np.abs(img.flatten()) - V))
    return sp.diags([1.0/(soft + 1e-5)], [0])


def make_D(N, M):
    u = sp.diags([[1] * (M - 1), [-1] * (M - 1)], [0, 1], shape=(M - 1, M))
    U = sp.block_diag([u] * N)

    V = sp.diags(
        [[1] * (M * (N - 1)), [-1] * (M * (N - 1))], [0, M],
        shape=(M * (N - 1), M * N))
    return sp.hstack([U.T, V.T]).T


# img: a 3 channels int image
# depth: a float 1 channel "image"
# Obviously img and depth must have the same 2D shape
def sharpen_depth(img, depth):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(float)

    M, N = img.shape

    D = make_D(M, N)

    d_img = D.dot(img.flatten())

    lam = 1e4
    lam2 = 5e-2
    V = 0.01
    #V = 2
    nb_iter = 300
    comp = soft_relu_comp_matrix

    C = sp.diags([0 + (np.random.rand(M * N) < 0.99)], [0])

    F = comp(d_img, V)
    A = sp.vstack([F.dot(D), lam * C])
    b = np.hstack([np.zeros((2 * M * N - M - N, )), (lam * C).dot(depth.flatten())])
    d = sp.linalg.lsqr(A, b, iter_lim=nb_iter)[0]

    showarray(d.reshape(M, N))

    for i in range(3):
        d2 = D.dot(d)

        G = comp(d2, V)

        A = sp.vstack([G.dot(D), lam2 * sp.identity(img.size)])
        b = np.hstack([np.zeros((2 * M * N - M - N, )), lam2 * img.flatten()])
        i = sp.linalg.lsqr(A, b, iter_lim=nb_iter)[0]

        showarray(i.reshape(M, N))
        C = sp.diags([0 + (np.random.rand(M * N) < 0.005)], [0])
        F2 = np.abs(D.dot(i))
        F2 = comp(F2, V)
        A = sp.vstack([F2.dot(D), lam * C])
        b = np.hstack([np.zeros((2 * M * N - M - N, )), (lam * C).dot(depth.flatten())])
        d = sp.linalg.lsqr(A, b, iter_lim=nb_iter)[0]

        showarray(d.reshape(M, N))
    return d.reshape(M, N)
