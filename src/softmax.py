import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
# 我们自定义线性回归函数看看在波士顿房价训练集上的效果
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
from func import softmax, accuracy


# 使用矩阵的逻辑回归
class softmax_cls:
    def __init__(self):
        self.w = None

    def fit(self, x, y, learning_rate=0.1, epochs: int = 10, L2=0.1):
        # 如果y只有一列，给他reshape一下，如果shape是(m,)容易出错，改为(m,1)
        if y.ndim < 2:
            raise Exception("y dims should be greater than 1")
        # 初始化w,目标函数，损失函数，求导，正则项，更新w
        # w是一个列向量。
        (m, k) = y.shape
        n = x.shape[1]
        np.random.seed(42)
        self.w = np.random.random((n, k))
        # 采用平方误差
        err = []
        for i in range(epochs):
            target = softmax(x, self.w)
            delta_regular = L2 * self.w
            # 向量化的梯度np.dot(x.T, (y - target))
            grad = -np.dot(x.T, (y - target)) / m
            grad = grad + delta_regular
            self.w = self.w - learning_rate * grad
            self.w[0, :] = np.sum(y - target, axis=0) / m
            err.append(loss(y, target, self.w))
        return self.w, err

    def predict(self, x):
        pred = softmax(x, self.w)
        return pred


def cls_plot(x, y, w):
    plt.scatter(x[:, 1], x[:, 2], c=y)
    x_range = [min(x[:, 1]), max(x[:, 1])]
    print(w[0], w[1], x_range, w[2])

    y = -(w[0] + np.multiply(w[1], x_range)) / w[2]
    plt.plot(x_range, y)
    plt.show()


def loss(y, y_pred, w):
    # y是独热编码后的m行k列矩阵，m个样本，k个类别
    m = y.shape[0]
    y_pred = np.clip(y_pred, 1e-5, 1 - 1e-5)
    regular = np.sum(np.power(w[1:, :], 2))
    # 前面是正规化项
    err = (-np.sum(np.multiply(y,np.log(y_pred)))-np.sum(np.multiply(1-y,np.log(1-y_pred))) + regular) / m
    return err


def one_hot_code(y):
    onehot = OneHotEncoder()
    y = y.reshape(-1, 1)
    y_one_hot = onehot.fit_transform(y).toarray()
    return y_one_hot


def creatdata():
    x, y = datasets.make_blobs(n_samples=1000, centers=3, cluster_std=0.1, random_state=1126,center_box=[-1,1])
    x=np.c_[x,x[:,0]**2,x[:,1]**2,x[:,0]*x[:,1]]
    return x, y


if __name__ == "__main__":
    # 载入数据，生成散点图
    x, y = creatdata()
    x = np.c_[np.ones(x.shape[0]), x]
    plt.scatter(x[:, 1], x[:, 2], c=y)
    plt.show()
    y = one_hot_code(y)

    # 划分数据集
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
    # 使用分类器
    cls = softmax_cls()
    y_val = np.argmax(y_val, axis=1)

    w, err = cls.fit(x_train, y_train, learning_rate=0.1, epochs=1000, L2=0.1)

    # 预测
    y_pred = cls.predict(x_val)
    y_pred = np.argmax(y_pred, axis=1)
    print(accuracy(y_val, y_pred))
    compare = (y_val == y_pred) * 1
    plt.scatter(x_val[:, 1], x_val[:, 2], c=y_val)
    plt.show()
    plt.scatter(x_val[:, 1], x_val[:, 2], c=compare)
    plt.legend(["1", "0"])
    plt.show()
    plt.plot(err)
    plt.show()
    temp = np.c_[y_val, y_pred]
    pass
