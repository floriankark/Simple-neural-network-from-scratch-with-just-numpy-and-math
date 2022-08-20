import numpy as np
import matplotlib.pyplot as plt

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
