import numpy as np

def relu(x):          return np.maximum(0, x)
def relu_prime(x):    return (x > 0).astype(x.dtype)

def sigmoid(x):       return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
def sigmoid_prime(x): return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):          return np.tanh(x)
def tanh_prime(x):    return 1 - np.tanh(x)**2

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)