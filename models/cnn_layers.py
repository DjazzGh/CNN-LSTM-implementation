import numpy as np

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """Converts an image into a column matrix. This is a common operation in CNNs
    to efficiently perform convolution as a matrix multiplication.

    Args:
        input_data (numpy.ndarray): Input data of shape (N, C, H, W)
                                    N: batch size, C: channels, H: height, W: width.
        filter_h (int): Height of the convolution filter.
        filter_w (int): Width of the convolution filter.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        pad (int, optional): Padding to apply to the input. Defaults to 0.

    Returns:
        numpy.ndarray: Column matrix of shape (N * out_h * out_w, C * filter_h * filter_w).
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """Converts a column matrix back into an image. This is used in the backward pass
    of convolutional layers to propagate gradients.

    Args:
        col (numpy.ndarray): Column matrix of shape (N * out_h * out_w, C * filter_h * filter_w).
        input_shape (tuple): Original input shape (N, C, H, W).
        filter_h (int): Height of the convolution filter.
        filter_w (int): Width of the convolution filter.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        pad (int, optional): Padding applied to the input. Defaults to 0.

    Returns:
        numpy.ndarray: Image data of shape (N, C, H, W).
    """
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


class Conv2D:
    """A 2D Convolutional Layer.

    Performs convolution operation on input data using learnable filters.
    Supports forward and backward passes, and stores gradients for optimization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad=0):
        """Initializes the Conv2D layer.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of filters (output channels) for the convolution.
            kernel_size (int): Size of the convolutional kernel (assumed square).
            stride (int, optional): Stride of the convolution. Defaults to 1.
            pad (int, optional): Padding to apply to the input. Defaults to 0.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

        # Initialize weights and biases
        # Weights are initialized using He initialization (Kaiming initialization)
        # for ReLU activation functions, which helps in preventing vanishing/exploding gradients.
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2. / (kernel_size * kernel_size * in_channels))
        self.bias = np.zeros((out_channels, 1)) # Bias is initialized to zeros

        # Gradients
        self.dweights = None
        self.dbias = None

        # Cache for backward pass
        self.col = None
        self.input_shape = None

    def forward(self, x):
        """Performs the forward pass of the Conv2D layer.

        Args:
            x (numpy.ndarray): Input data of shape (N, C, H, W).

        Returns:
            numpy.ndarray: Output feature map after convolution.
        """
        self.input_shape = x.shape
        N, C, H, W = x.shape
        out_h = (H + 2 * self.pad - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.pad - self.kernel_size) // self.stride + 1

        # Reshape input and weights for efficient matrix multiplication
        col = im2col(x, self.kernel_size, self.kernel_size, self.stride, self.pad)
        col_W = self.weights.reshape(self.out_channels, -1).T

        # Perform convolution as matrix multiplication
        out = np.dot(col, col_W) + self.bias.T
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.col = col # Cache for backward pass
        return out

    def backward(self, dout):
        """Performs the backward pass of the Conv2D layer.

        Computes gradients with respect to weights, biases, and input data.

        Args:
            dout (numpy.ndarray): Gradient from the subsequent layer.

        Returns:
            numpy.ndarray: Gradient with respect to the input data.
        """
        N, C, H, W = self.input_shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

        # Compute gradients for bias and weights
        self.dbias = np.sum(dout, axis=0).reshape(self.out_channels, 1)
        self.dweights = np.dot(self.col.T, dout).transpose(1, 0).reshape(
            self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)

        # Compute gradient with respect to input data
        dcol = np.dot(dout, self.weights.reshape(self.out_channels, -1))
        dx = col2im(dcol, self.input_shape, self.kernel_size, self.kernel_size, self.stride, self.pad)
        return dx

    def parameters(self):
        """Returns the learnable parameters of the layer.

        Returns:
            dict: A dictionary containing 'weights' and 'bias'.
        """
        return {'weights': self.weights, 'bias': self.bias}

    def grads(self):
        """Returns the gradients of the learnable parameters.

        Returns:
            dict: A dictionary containing 'dweights' and 'dbias'.
        """
        return {'dweights': self.dweights, 'dbias': self.dbias}


class MaxPool2D:
    """A 2D Max Pooling Layer.

    Downsamples the input feature map by taking the maximum value within each pooling window.
    """
    def __init__(self, pool_size, stride=None):
        """Initializes the MaxPool2D layer.

        Args:
            pool_size (int): Size of the pooling window (assumed square).
            stride (int, optional): Stride of the pooling operation. Defaults to pool_size.
        """
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size

        # Cache for backward pass
        self.x = None
        self.arg_max = None

    def forward(self, x):
        """Performs the forward pass of the MaxPool2D layer.

        Args:
            x (numpy.ndarray): Input data of shape (N, C, H, W).

        Returns:
            numpy.ndarray: Output feature map after max pooling.
        """
        self.x = x
        N, C, H, W = x.shape
        out_h = (H - self.pool_size) // self.stride + 1
        out_w = (W - self.pool_size) // self.stride + 1

        # Reshape input for efficient pooling
        col = im2col(x, self.pool_size, self.pool_size, self.stride, pad=0)
        col = col.reshape(-1, self.pool_size * self.pool_size)

        # Perform max pooling
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.arg_max = arg_max # Cache for backward pass
        return out

    def backward(self, dout):
        """Performs the backward pass of the MaxPool2D layer.

        Propagates gradients by placing them at the positions of the maximum values
        found during the forward pass.

        Args:
            dout (numpy.ndarray): Gradient from the subsequent layer.

        Returns:
            numpy.ndarray: Gradient with respect to the input data.
        """
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_size * self.pool_size
        dmax = np.zeros((dout.size, pool_size))
        # Place gradients at the positions of the maximum values
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        # Convert column matrix of gradients back to image shape
        dx = col2im(dmax, self.x.shape, self.pool_size, self.pool_size, self.stride, pad=0)
        return dx


class Flatten:
    """A Flatten Layer.

    Reshapes the input tensor into a 2D tensor (batch_size, -1),
    where -1 means the dimension is inferred from the other dimensions.
    """
    def __init__(self):
        """Initializes the Flatten layer.
        """
        self.input_shape = None

    def forward(self, x):
        """Performs the forward pass of the Flatten layer.

        Args:
            x (numpy.ndarray): Input data of arbitrary shape.

        Returns:
            numpy.ndarray: Flattened output of shape (batch_size, -1).
        """
        self.input_shape = x.shape
        # Reshape to (batch_size, -1) where -1 infers the dimension
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        """Performs the backward pass of the Flatten layer.

        Reshapes the gradient back to the original input shape.

        Args:
            dout (numpy.ndarray): Gradient from the subsequent layer, shape (batch_size, -1).

        Returns:
            numpy.ndarray: Gradient reshaped to the original input shape.
        """
        # Reshape the gradient back to the original input shape
        return dout.reshape(self.input_shape)