import numpy as np


def relu(z):
    z[z <= 0] = 0
    return z

def linear(z):
    return z

def sigmoid(z):
    return 1 / ( 1 + np.exp(-z))