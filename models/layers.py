import numpy as np


class Linear:
    """A fully connected (dense) layer in a neural network.

    This layer performs a linear transformation on its input: Y = XW^T + b.
    It stores weights (W) and biases (b) and computes their gradients during backpropagation.
    """
    def __init__(self, input_size, output_size):
        """Initializes the Linear layer with weights and biases.

        Args:
            input_size (int): The number of input features.
            output_size (int): The number of output features.
        """
        # Initialize weights with a small random normal distribution
        self.W = np.random.randn(output_size, input_size) * 0.01
        # Initialize biases to zeros
        self.b = np.zeros((output_size, 1))

        # Caches for backward pass
        self.X = None

        # Gradients
        self.dW = None
        self.db = None

    def forward(self, X):
        """Performs the forward pass for the Linear layer.

        Args:
            X (numpy.ndarray): Input data of shape (batch_size, input_size).

        Returns:
            numpy.ndarray: Output of the linear transformation (batch_size, output_size).
        """
        self.X = X.T # Cache input for backward pass, transpose to (input_size, batch_size)
        # Compute Y = WX + b
        output = np.dot(self.W, self.X) + self.b
        return output.T # Transpose back to (batch_size, output_size)

    def backward(self, d_output):
        """Performs the backward pass for the Linear layer.

        Computes gradients with respect to weights, biases, and input.

        Args:
            d_output (numpy.ndarray): Gradient of the loss with respect to the output of this layer
                                      (batch_size, output_size).

        Returns:
            numpy.ndarray: Gradient of the loss with respect to the input of this layer
                           (batch_size, input_size).
        """
        d_output = d_output.T # Transpose gradient to (output_size, batch_size)

        # Compute gradients for weights and biases
        self.dW = np.dot(d_output, self.X.T)
        self.db = np.sum(d_output, axis=1, keepdims=True)

        # Compute gradient with respect to the input
        d_input = np.dot(self.W.T, d_output)
        return d_input.T # Transpose back to (batch_size, input_size)

    def parameters(self):
        """Returns the learnable parameters of the Linear layer.

        Returns:
            list: A list containing the weight matrix (W) and bias vector (b).
        """
        return [self.W, self.b]

    def grads(self):
        """Returns the gradients of the learnable parameters of the Linear layer.

        Returns:
            list: A list containing the gradients for the weight matrix (dW) and bias vector (db).
        """
        return [self.dW, self.db]