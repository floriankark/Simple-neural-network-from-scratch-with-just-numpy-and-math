import numpy as np
import matplotlib.pyplot as plt

def generate_data(n):
    X1 = np.random.multivariate_normal([1,2], [[1,0], [0,1]], int(n/2))
    X2 = np.random.multivariate_normal([3,5], [[1,0], [0,1]], int(n/2))
    X = np.concatenate((X1, X2))
    y = np.concatenate((np.ones(int(n/2)), -np.ones(int(n/2))))
    return X, y

X_train, y_train = generate_data(200)
X_test, y_test = generate_data(100)

scatter = plt.scatter(X_train[:,0], X_train[:,1], c=y_train)
#plt.scatter(X_test[:,0], X_test[:,1], c=y_test)
plt.legend(*scatter.legend_elements(), loc=4)
plt.show()
