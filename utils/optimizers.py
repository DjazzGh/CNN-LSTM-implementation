import numpy as np

class SGD:
    def __init__(self, lr=0.01, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def step(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]
        for i, (p, g, v) in enumerate(zip(params, grads, self.v)):
            v[:] = self.momentum * v + g
            p -= self.lr * v
        return params