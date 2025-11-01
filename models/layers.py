import numpy as np
from utils.activations import relu, relu_prime

class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros((1, out_features))
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        dW = self.x.T @ dout
        db = np.sum(dout, axis=0, keepdims=True)
        dx = dout @ self.W.T
        self.dW = dW
        self.db = db.T
        return dx, self.dW, self.db

    def parameters(self):
        return [self.W, self.b]

    def grads(self):
        return {'dW': self.dW, 'db': self.db}