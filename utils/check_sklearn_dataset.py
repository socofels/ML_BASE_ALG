from sklearn import datasets
from matplotlib import pyplot as plt

x,y = datasets.make_regression(n_features=1,noise=50)
plt.scatter(x[:,0],y)
plt.show()
print(x.shape,y.shape)
