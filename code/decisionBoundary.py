import numpy as np
import matplotlib.pyplot as plt

#now only shows test data
X = X_test #np.concatenate((X_train, X_test)) #would show all data
y = X_test #np.concatenate((y_train, y_test))

x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.05), np.arange(x2_min, x2_max, 0.05))

#Plot the decision boundary. Therefore assign a color to each
#point in the mesh [x_min, m_max]x[y_min, y_max].
X_grid = np.c_[xx1.ravel(), xx2.ravel()]
y_grid = net.output(X_grid).reshape(xx1.shape)

#Put the result into a color plot and set levels to zeros
#to get a separation into two classes
plt.contourf(xx1, xx2, y_grid, levels=0, cmap=plt.cm.bwr)
#Plot the test points and the true labels
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr)

plt.title('Decision boundary')
plt.axis("off")
#plt.colorbar() #shows a colorbar on the right hand sight of the plot
plt.show()
