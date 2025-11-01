import numpy as np
from .activations import softmax

def cross_entropy(logits, y_onehot):
    probs = softmax(logits, axis=1)
    return -np.mean(np.sum(y_onehot * np.log(probs + 1e-12), axis=1))

def cross_entropy_prime(logits, y_onehot):
    probs = softmax(logits, axis=1)
    return probs - y_onehot