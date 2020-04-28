import numpy as np
from matplotlib import pyplot as plt


def sign(y_pred):
    y_pred = (y_pred >= 0) * 2 - 1
    return y_pred


def plot(x, w):
    plt.scatter(x[:, 1][pos_index], x[:, 2][pos_index], marker="P")
    plt.scatter(x[:, 1][neg_index], x[:, 2][neg_index], marker=0)
    x = [-1, 100]
    y = -(w[0] + w[1] * x) / w[2]
    plt.plot(x, y)
    plt.show()

# 当没有最优解时，循环一百次后结束
def pla(x, y,epochs=100):
    w = np.random.random((x.shape[1], 1))
    plot(x, w)
    best_w =w
    for i in range(epochs):
        if not (sign(y)==sign(np.dot(x, w))).all():
            for index in np.where(sign(y)!=sign(np.dot(x, w)))[0]:
                y_pred = sign(np.dot(x[index,:], w))
                y_true = sign(y[index])
                # 如果预测的y与实际的y不相等，就更新w。y*x
                if not y_pred == y_true:
                    temp = x[index,:] * y[index]
                    temp = temp.reshape(-1, 1)
                    w = w + temp
                    plot(x, w)
            if np.sum(sign(y) == sign(np.dot(x, w))) > np.sum(sign(y) == sign(np.dot(x, best_w))):
                best_w = w
        else:
            break
    return best_w


np.random.seed(3)
shape = (100, 2)
x = (np.random.random((shape[0], shape[1])) * 100).astype(int)
x = np.c_[np.ones((shape[0], 1)), x]
w = np.array([-5, -2, 2])
w = w.reshape(-1, 1)
y = np.dot(x, w)
pos_index = np.where(y > 10)[0]
neg_index = np.where(y < 10)[0]
plt.scatter(x[:, 1][pos_index], x[:, 2][pos_index], marker="P")
plt.scatter(x[:, 1][neg_index], x[:, 2][neg_index], marker=0)
plt.show()
best_w = pla(x, y,100)
print(best_w)
