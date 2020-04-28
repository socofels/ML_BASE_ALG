import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split


# 我们自定义线性回归函数看看在波士顿房价训练集上的效果

class lin_reg:
    def __init__(self):
        self.w = None
    def fit(self, x, y, learning_rate=0.001, epochs: int = 10, L2=0.1):
        # 如果y只有一列，给他reshape一下，如果shape是(m,)容易出错，改为(m,1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        # 初始化w,目标函数，损失函数，求导，正则项，更新w
        # w是一个列向量。
        m = len(y)
        self.w = np.random.random((x.shape[1], 1))
        target = np.dot(x, self.w)
        # 采用平方误差
        for i in range(epochs):
            y_hat = np.dot(x, self.w)  # (m,1)
            loss = np.sum(np.power((y - np.dot(x, self.w)), 2)) / m
            regular = self.w / m * L2
            grad = 2 * np.dot(x.T, np.dot(x, self.w) - y) / m + regular
            w0 = np.sum(y - y_hat) / m
            self.w = self.w - grad*learning_rate
            self.w[0] = w0
            print(f"loss:{loss},grad:{grad[1]},b:{self.w[0]},regular:{regular[1]},w:{self.w[0],self.w[1]}-----lin_reg")
    def predict(self, x):
        pred = np.dot(x, self.w)
        return pred

if __name__ == "__main__":
    # 划分训练集和验证集合
    x, y = datasets.make_regression(n_features=1, noise=50, n_samples=10, random_state=99)
    x = np.c_[np.ones(x.shape[0]), x]
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
    # 使用自定义线性回归训练
    lin_b = lin_reg()
    lin_b.fit(x_train, y_train, learning_rate=0.001, epochs=1000, L2=0.1)
    y_pred = lin_b.predict(x)
    plt.scatter(x[:, 1], y)
    plt.plot(x[:, 1], y_pred)
    plt.show()
    pass
