import numpy as np

from models.cnn_layers import Conv2D, MaxPool2D, Flatten
from models.lstm_layers import LSTM
from models.layers import Linear


class CNNLSTM:
    """Combines CNN layers for feature extraction with an LSTM layer for sequence processing.

    The CNN part processes the input image to extract spatial features, which are then
    flattened and fed into the LSTM layer. The LSTM processes these features as a sequence
    to capture temporal dependencies, and a final linear layer outputs the classification scores.
    """
    def __init__(self, input_shape, num_classes, cnn_filters, cnn_kernel_sizes,
                 cnn_strides, cnn_paddings, pool_sizes, lstm_hidden_size,
                 lstm_sequence_length):
        """Initializes the CNN-LSTM model with specified architecture parameters.

        Args:
            input_shape (tuple): Shape of the input data (channels, height, width).
            num_classes (int): Number of output classes for classification.
            cnn_filters (list): List of integers, number of filters for each Conv2D layer.
            cnn_kernel_sizes (list): List of integers, kernel size for each Conv2D layer.
            cnn_strides (list): List of integers, stride for each Conv2D layer.
            cnn_paddings (list): List of integers, padding for each Conv2D layer.
            pool_sizes (list): List of integers, pool size for each MaxPool2D layer.
            lstm_hidden_size (int): Number of hidden units in the LSTM layer.
            lstm_sequence_length (int): The length of the sequence fed into the LSTM.
                                        This is typically the flattened output size of the CNN part.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lstm_sequence_length = lstm_sequence_length

        # CNN Layers
        self.conv1 = Conv2D(input_shape[0], cnn_filters[0], cnn_kernel_sizes[0],
                            cnn_strides[0], cnn_paddings[0])
        self.pool1 = MaxPool2D(pool_sizes[0])
        self.conv2 = Conv2D(cnn_filters[0], cnn_filters[1], cnn_kernel_sizes[1],
                            cnn_strides[1], cnn_paddings[1])
        self.pool2 = MaxPool2D(pool_sizes[1])
        self.flatten = Flatten() # Flattens the output of CNN to feed into LSTM

        # Calculate the input size for the LSTM layer after CNN and flattening
        # This involves a dummy forward pass through the CNN part to determine the output shape.
        dummy_input = np.zeros((1,) + input_shape)
        dummy_output = self.pool2.forward(self.conv2.forward(self.pool1.forward(self.conv1.forward(dummy_input))))
        self.lstm_input_size = dummy_output.shape[1] * dummy_output.shape[2] * dummy_output.shape[3] // lstm_sequence_length

        # LSTM Layer
        self.lstm = LSTM(self.lstm_input_size, lstm_hidden_size, lstm_sequence_length)

        # Output Linear Layer
        self.linear = Linear(lstm_hidden_size, num_classes) # Maps LSTM output to class scores

        self.layers = [self.conv1, self.pool1, self.conv2, self.pool2, self.flatten, self.lstm, self.linear]

    def forward(self, x):
        """Performs the forward pass through the CNN-LSTM model.

        Args:
            x (numpy.ndarray): Input data, typically a batch of images.

        Returns:
            numpy.ndarray: Output logits (raw scores) for each class.
        """
        # CNN Forward Pass
        x = self.conv1.forward(x)
        x = self.pool1.forward(x)
        x = self.conv2.forward(x)
        x = self.pool2.forward(x)
        x = self.flatten.forward(x)

        # Reshape for LSTM: (batch_size, sequence_length, input_size_per_step)
        # The flattened CNN output is reshaped to be a sequence for the LSTM.
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.lstm_sequence_length, self.lstm_input_size)

        # LSTM Forward Pass
        x = self.lstm.forward(x)

        # Linear Layer Forward Pass (using the last hidden state of LSTM)
        x = self.linear.forward(x)
        return x

    def backward(self, grad):
        """Performs the backward pass through the CNN-LSTM model to compute gradients.

        Args:
            grad (numpy.ndarray): Gradient of the loss with respect to the model's output.

        Returns:
            numpy.ndarray: Gradient of the loss with respect to the model's input.
        """
        # Linear Layer Backward Pass
        grad = self.linear.backward(grad)

        # LSTM Backward Pass
        # Reshape the gradient from (batch_size, lstm_hidden_size) to (batch_size, lstm_sequence_length, lstm_hidden_size)
        # to match the LSTM's output shape during forward pass for proper backpropagation.
        batch_size = grad.shape[0]
        grad = grad.reshape(batch_size, self.lstm_sequence_length, self.lstm.hidden_size)
        grad = self.lstm.backward(grad)

        # Reshape gradient for Flatten layer backward pass
        # The gradient from LSTM needs to be reshaped back to the CNN output shape
        # before it can be processed by the Flatten layer's backward method.
        grad = grad.reshape(batch_size, -1)

        # CNN Backward Pass
        grad = self.flatten.backward(grad)
        grad = self.pool2.backward(grad)
        grad = self.conv2.backward(grad)
        grad = self.pool1.backward(grad)
        grad = self.conv1.backward(grad)
        return grad

    def parameters(self):
        """Returns a list of all learnable parameters in the model.

        Returns:
            list: A list of dictionaries, where each dictionary contains 'weights' and 'bias' for a layer.
        """
        params = []
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                params.append({'weights': layer.weights, 'bias': layer.bias})
            elif hasattr(layer, 'Wf'): # For LSTM layer
                params.append({
                    'Wf': layer.Wf, 'Wi': layer.Wi, 'Wo': layer.Wo, 'Wg': layer.Wg,
                    'bf': layer.bf, 'bi': layer.bi, 'bo': layer.bo, 'bg': layer.bg
                })
        return params

    def grads(self):
        """Returns a list of gradients for all learnable parameters in the model.

        Returns:
            list: A list of dictionaries, where each dictionary contains 'weights' and 'bias' gradients for a layer.
        """
        grads = []
        for layer in self.layers:
            if hasattr(layer, 'dweights'):
                grads.append({'weights': layer.dweights, 'bias': layer.dbias})
            elif hasattr(layer, 'dWf'): # For LSTM layer
                grads.append({
                    'dWf': layer.dWf, 'dWi': layer.dWi, 'dWo': layer.dWo, 'dWg': layer.dWg,
                    'dbf': layer.dbf, 'dbi': layer.dbi, 'dbo': layer.dbo, 'dbg': layer.dbg
                })
        return grads