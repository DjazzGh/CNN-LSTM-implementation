import numpy as np
from utils.activations import softmax

def cross_entropy(y_true, y_pred):
    """Calculates the cross-entropy loss.

    Args:
        y_true (numpy.ndarray): True labels (one-hot encoded).
        y_pred (numpy.ndarray): Predicted probabilities (output of softmax).

    Returns:
        float: The computed cross-entropy loss.
    """
    # Clip predictions to avoid log(0) errors
    y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
    return -np.sum(y_true * np.log(y_pred))

def cross_entropy_prime(y_true, y_pred):
    """Calculates the derivative of the cross-entropy loss with respect to the predictions.

    Args:
        y_true (numpy.ndarray): True labels (one-hot encoded).
        y_pred (numpy.ndarray): Predicted probabilities (output of softmax).

    Returns:
        numpy.ndarray: The gradient of the cross-entropy loss with respect to y_pred.
    """
    # The derivative of cross-entropy loss with softmax is simply y_pred - y_true
    return y_pred - y_true