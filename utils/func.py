import numpy as np


def sigmoid(x):
    a = np.power(np.e, -x)
    return 1 / (1 + a)


def sign(x):
    x = (x >= 0.5) * 1
    return x


def softmax(x, w):
    # x是m行n列矩阵
    # w是n行k列矩阵
    # z是m行k列矩阵
    # 优化它
    w0=w[0:,:]
    a=np.max(w,axis=1,keepdims=True)
    w=w-np.max(w,axis=1,keepdims=True)
    # w[0:,:]=w0
    z = np.dot(x, w)
    g_z = np.exp(z)
    # 求每个数据的g(z)总和
    z_sum = np.sum(g_z, axis=1, keepdims=True)
    prob = g_z/z_sum
    # 返回的概率是m*k矩阵
    # prob = np.clip(prob,1e-5,1-1e-5)
    return prob


def accuracy(y, y_pred):
    reslut = np.sum((y == y_pred))
    return reslut / y.shape[0]
