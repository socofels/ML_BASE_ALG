import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from func import sigmoid, sign
from sklearn.preprocessing import StandardScaler
from time import time
from configs.color import cnames
# 我们自定义线性回归函数看看在波士顿房价训练集上的效果
from sklearn.linear_model import LogisticRegression


# 使用矩阵的逻辑回归
class logistic_cls:
    def __init__(self):
        self.w = None

    def fit(self, x, y, learning_rate=0.01, epochs: int = 10, L2=0.1):
        # 如果y只有一列，给他reshape一下，如果shape是(m,)容易出错，改为(m,1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        # 初始化w,目标函数，损失函数，求导，正则项，更新w
        # w是一个列向量。
        m = len(y)
        np.random.seed(42)
        self.w = np.random.random((x.shape[1], 1))
        # 采用平方误差
        for i in range(epochs):
            target = np.clip(sigmoid(np.dot(x, self.w)), 1e-5, 1 - 1e-5)
            grad = np.dot(x.T, target - target ** 2 * y) / m + self.w * L2
            loss = - (np.dot(y.T, np.log(target)) + np.dot(1 - y.T, np.log(1 - target))) / m
            self.w = self.w - learning_rate * grad
            self.w[0] = np.sum(y - target) / m
            print(f"loss:{loss}，grad{grad.T}")
        return self.w

    def predict(self, x):
        pred = np.dot(x, self.w)
        return pred


# 使用for循环的逻辑回归
class logistic_clsa:
    def __init__(self):
        self.w = None

    def fit(self, x, y, learning_rate=0.01, epochs: int = 10, L2=0.1):
        # 如果y只有一列，给他reshape一下，如果shape是(m,)容易出错，改为(m,1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        # 初始化w,目标函数，损失函数，求导，正则项，更新w
        # w是一个列向量。
        m = len(y)
        np.random.seed(42)
        self.w = np.random.random((x.shape[1], 1))
        # 采用平方误差
        y_pred = np.zeros(y.shape)
        grad = 0
        b = 0
        for i in range(epochs):
            for j in range(m):
                y_pred[j] = sigmoid(np.dot(x[j, :], self.w))
                temp = (1 - y_pred[j]) * y_pred[j] * x[j, :] * y[j] + (1 - y[j]) * y_pred[j] * x[j, :]
                grad = temp.reshape(-1, 1) + grad
                b = b + y[j] - y_pred[j]
            regular = L2 / m * self.w
            grad = grad.reshape(-1, 1)
            grad = grad / m + regular
            b = b / m
            self.w = self.w - learning_rate * grad
            self.w[0] = b
            err = loss(y, y_pred, w)
            print(f"loss:{err}，grad{grad.T},,,a")
        return self.w

    def predict(self, x):
        pred = np.dot(x, self.w)
        return pred


def cls_plot(x, y, w):
    plt.scatter(x[:, 1], x[:, 2], c=y)
    x_range = [min(x[:, 1]), max(x[:, 1])]
    y = -(w[0] + w[1] * x_range) / w[2]
    plt.plot(x_range, y)
    plt.show()


def loss(y, y_pred, w):
    regular = np.sum(np.power(w[1:, :], 2))
    m = len(y)
    y_pred = np.clip(y_pred, 1e-5, 1 - 1e-5)
    err = (-np.dot(y.T, np.log(y_pred)) - np.dot((1 - y).T, np.log(1 - y_pred)) + regular) / m
    return err


if __name__ == "__main__":
    # 划分训练集和验证集合
    x, y = datasets.make_blobs(centers=2, cluster_std=1, random_state=1126)
    stand = StandardScaler()
    x = stand.fit_transform(x)
    x = np.c_[np.ones(x.shape[0]), x]
    plt.scatter(x[:, 1], x[:, 2], c=y)
    plt.show()
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
    # # 使用自定义逻辑回归训练
    cls = logistic_cls()
    start = time()
    w = cls.fit(x_train, y_train, learning_rate=0.1, epochs=5, L2=0.1)
    time_a = time()
    ww = logistic_clsa().fit(x_train, y_train, learning_rate=1, epochs=5, L2=0.1)
    time_b = time()
    print(f"矩阵用时{time_a - start},for循环用时{time_b - time_a}")
    y_pred = cls.predict(x)
    cls_plot(x, y, w)
    # plt.plot(x[:, 1], y_pred)
    # plt.show()
    pass
    # y=np.array([0,1,1])
    # y_pred = np.array([0.1,0.8,0])
    # print(loss(y,y_pred))
