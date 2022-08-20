import numpy as np

#Mean squared error loss
def mse(y_true, output):
    return np.mean(np.power(y_true-output, 2))
def dmse(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

#Hinge loss
def hinge_loss(y_true, output):
    return np.mean(np.maximum(0, 1 - y_true*output))
def dhinge_loss(y_true, output):
    return -y_true * (1 >= y_true * output)
