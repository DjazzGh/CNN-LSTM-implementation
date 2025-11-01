import numpy as np

def relu(x):
    """Rectified Linear Unit (ReLU) activation function.
    
    Args:
        x (numpy.ndarray): Input array.
    Returns:
        numpy.ndarray: Output array with ReLU applied.
    """
    return np.maximum(0, x)

def relu_prime(x):
    """Derivative of the ReLU activation function.
    
    Args:
        x (numpy.ndarray): Input array.
    Returns:
        numpy.ndarray: Derivative of ReLU applied to the input.
    """
    return (x > 0).astype(float)

def sigmoid(x):
    """Sigmoid activation function.
    
    Args:
        x (numpy.ndarray): Input array.
    Returns:
        numpy.ndarray: Output array with Sigmoid applied.
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    """Derivative of the Sigmoid activation function.
    
    Args:
        x (numpy.ndarray): Input array.
    Returns:
        numpy.ndarray: Derivative of Sigmoid applied to the input.
    """
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    """Hyperbolic Tangent (tanh) activation function.
    
    Args:
        x (numpy.ndarray): Input array.
    Returns:
        numpy.ndarray: Output array with Tanh applied.
    """
    return np.tanh(x)

def tanh_prime(x):
    """Derivative of the Tanh activation function.
    
    Args:
        x (numpy.ndarray): Input array.
    Returns:
        numpy.ndarray: Derivative of Tanh applied to the input.
    """
    return 1 - np.tanh(x)**2

def softmax(x):
    """Softmax activation function.
    
    Args:
        x (numpy.ndarray): Input array.
    Returns:
        numpy.ndarray: Output array with Softmax applied.
    """
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True)) # For numerical stability
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)