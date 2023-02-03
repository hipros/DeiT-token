import math
import numpy as np
from main import Hook


def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n

    return np.dot(np.dot(H, K), H)  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
    # return np.dot(H, K)  # KH


def rbf(X, sigma=None):
    GX = np.dot(X, X.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = np.exp(KX)
    return KX


def kernel_HSIC(X, Y, sigma):
    return np.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))


def linear_HSIC(X, Y):
    L_X = np.dot(X, X.T)
    L_Y = np.dot(Y, Y.T)
    return np.sum(centering(L_X) * centering(L_Y))


def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = np.sqrt(linear_HSIC(X, X))
    var2 = np.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)


def kernel_CKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = np.sqrt(kernel_HSIC(X, X, sigma))
    var2 = np.sqrt(kernel_HSIC(Y, Y, sigma))

    return hsic / (var1 * var2)


if __name__=='__main__':
    
    print(Hook.cls_array)
    X = np.array([[ 1.0830,  2.7031,  1.9814, -2.0840, -1.8945, -2.1270],
         [ 1.6514,  9.5312,  2.2637, -3.3125, -2.1152, -2.2637],
         [ 1.6143,  6.3672,  2.2832, -3.0957, -1.8027, -2.5039],
         [ 1.5391,  6.4922,  1.7188, -4.5508, -1.8369, -2.1660],
         [ 0.8579,  3.7891,  1.7803, -1.0449, -1.7334, -1.8311],
         [ 1.8818,  8.2188,  2.1387, -4.9727, -1.9229, -2.4746]])

    Y = np.array([[ 2.6250, -2.2305,  0.1891, -1.4609, -2.2500, -1.8965],
        [ 3.0195,  5.4375,  0.5063, -0.5459, -2.8887, -1.4014],
        [ 3.1387,  0.8340,  0.4695, -0.9912, -2.2930, -1.9053],
        [ 3.2402,  1.8262,  0.4207, -2.8418, -2.6230, -2.0488],
        [ 2.3867,  1.0381,  0.0905,  1.1865, -1.9609, -2.2031],
        [ 3.1543,  0.4258,  0.3035, -2.5469, -2.3379, -1.5781]])

    print('Linear CKA, between X and Y: {}'.format(linear_CKA(X, Y)))
    print('Linear CKA, between X and X: {}'.format(linear_CKA(X, X)))

    print('RBF Kernel CKA, between X and Y: {}'.format(kernel_CKA(X, Y)))
    print('RBF Kernel CKA, between X and X: {}'.format(kernel_CKA(X, X)))