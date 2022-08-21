import numpy as np
import matplotlib.pyplot as plt

#I used seed 100 for this project so you can recreate it
np.random.seed(100) 

#---GENERATE AND PLOT TOY DATA---
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


#---IMPLEMENTATION OF MODEL---
class linearNet:
    
    def __init__(self, d):
        self.w = np.random.normal(scale = np.sqrt(d), size=(d,1))
        self.b = 0

    def output(self, X):
        return X @ self.w + self.b

    def grad(self, X, y_true, dloss_function):
        output = self.output(X)
        y_true = y_true.reshape(-1,1)
        dloss = dloss_function(y_true, output)

        grad_w = (dloss * X).mean(axis=0).reshape(-1,1)
        grad_b = dloss.mean()

        return grad_w, grad_b
    
    def fit(self, network, X_train, y_train, epochs, learning_rate, loss_function, dloss_function):
        #used for plotting later
        loss_hist = []
        accuracy_hist = []

        for i in range(epochs):
            grad_w, grad_b = net.grad(X_train, y_train, dloss_function)

            network.w -= learning_rate * grad_w
            network.b -= learning_rate * grad_b

            output = network.output(X_train)[:,0]
            
            #calculate loss and accuracy for plotting
            loss = loss_function(y_train, output)
            accuracy = compute_accuracy(y_train, output)

            loss_hist.append(loss)
            accuracy_hist.append(accuracy)

        #plot loss and accuracy on train set 
        plt.title('Loss history')
        plt.plot(loss_hist)
        plt.show()
        print("Most recent loss is", loss)

        plt.title('Accuracy history')
        plt.plot(accuracy_hist)
        plt.show()
        print("Most recent accuracy is "+str(accuracy*100)+"%")

#---LOSS---
def mse(y_true, output):
    return np.mean(np.power(output-y_true, 2))/y_true.size
def dmse(y_true, output):
    return 2*(output-y_true)/y_true.size

#---ACCURACY---
def compute_accuracy(y_true, output):
    #negative values will be labeled -1 and positive ones 1
    y_pred = np.sign(output)
    return (y_true == y_pred).mean()

#---TRAINING---
#d=2 because we got inputs as 2 dimensional coordinates
net = linearNet(2)
net.fit(net, X_train, y_train, epochs=10000, learning_rate=0.1, loss_function=mse, dloss_function=dmse)

#---PLOT TRAINING RESULTS---
X = X_test #np.concatenate((X_train, X_test))
y = y_test #np.concatenate((y_train, y_test))

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
#plt.colorbar()
plt.show()

test_output = net.output(X_test)[:,0]
print("The accuracy on test data is "+str(compute_accuracy(y_test, test_output)*100)+"%")
