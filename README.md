<h1 align="center">Creating a simple neural network from scratch</h1>

## Description
This is the first part of a series of projects that will eventually end in a self made library to build neural networks with a variety of layers so that I will be able to have a neural network that solves MNIST or even more complicated problems. But first I start at the very beginning, a single layer linear neural network that solves a simple classification problem. 

## The Problem
Let's start by having a look at the data and the problem we want to solve. Therefore I generate two bivariat normal distributed point clouds X1 and X2 and give each of them a label stored in y.

``` bash
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

plt.scatter(X_train[:,0], X_train[:,1], c=y_train)
scatter = plt.scatter(X_test[:,0], X_test[:,1], c=y_test)
plt.legend(*scatter.legend_elements(), loc=4)
plt.show()
```
<p align="center"> 
    <img width=400 src="./visualization/generatedData.png" alt="generated data">
</p>
