# models/cnn_layers.py
import numpy as np
from utils.activations import relu, relu_prime


# ----------------------------------------------------------------------
#  im2col / col2im utilities
# ----------------------------------------------------------------------
def im2col(x, f_h, f_w, stride, pad):
    """
    Transform a 4‑D tensor into columns.
    Input : (N, C, H, W)
    Output: (N*OH*OW, C*f_h*f_w)
    """
    N, C, H, W = x.shape
    H_p = H + 2 * pad
    W_p = W + 2 * pad
    x_padded = np.pad(x,
                      ((0, 0), (0, 0), (pad, pad), (pad, pad)),
                      mode='constant')

    OH = (H_p - f_h) // stride + 1
    OW = (W_p - f_w) // stride + 1

    cols = np.zeros((N * OH * OW, C * f_h * f_w), dtype=x.dtype)

    idx = 0
    for n in range(N):
        for i in range(OH):
            for j in range(OW):
                patch = x_padded[
                    n,
                    :,
                    i * stride : i * stride + f_h,
                    j * stride : j * stride + f_w,
                ]
                cols[idx] = patch.ravel()
                idx += 1
    return cols, (N, C, H, W, OH, OW, stride, pad)


def col2im(cols, shape_info, f_h, f_w):
    """
    Reverse of im2col.
    """
    N, C, H, W, OH, OW, stride, pad = shape_info
    H_p = H + 2 * pad
    W_p = W + 2 * pad
    dx_padded = np.zeros((N, C, H_p, W_p), dtype=cols.dtype)

    idx = 0
    for n in range(N):
        for i in range(OH):
            for j in range(OW):
                patch = cols[idx].reshape(C, f_h, f_w)
                dx_padded[
                    n,
                    :,
                    i * stride : i * stride + f_h,
                    j * stride : j * stride + f_w,
                ] += patch
                idx += 1

    # remove padding
    if pad > 0:
        dx = dx_padded[:, :, pad:-pad, pad:-pad]
    else:
        dx = dx_padded
    return dx


# ----------------------------------------------------------------------
#  Conv2D
# ----------------------------------------------------------------------
class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, pad=0):
        k = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        limit = np.sqrt(6 / (in_channels * k * k))
        self.W = np.random.uniform(-limit, limit,
                                   (out_channels, in_channels, k, k))
        self.b = np.zeros((out_channels, 1))

        self.stride = stride
        self.pad = pad

        # caches
        self.x_col = None
        self.W_col = None
        self.shape_info = None
        self.x_padded = None
        self.last_x = None          # for ReLU derivative

    def forward(self, x):
        """
        x : (N, C, H, W)
        """
        self.last_x = x
        N, C, H, W = x.shape
        F, _, HH, WW = self.W.shape

        OH = (H + 2 * self.pad - HH) // self.stride + 1
        OW = (W + 2 * self.pad - WW) // self.stride + 1

        x_padded = np.pad(x,
                          ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)),
                          mode='constant')
        self.x_padded = x_padded

        self.x_col, self.shape_info = im2col(x_padded, HH, WW,
                                            self.stride, 0)
        self.W_col = self.W.reshape(F, -1)

        pre_activation_out = self.x_col @ self.W_col.T + self.b.T          # (N*OH*OW, F)
        out = pre_activation_out.reshape(N, OH, OW, F).transpose(0, 3, 1, 2) # (N, F, OH, OW)
        self.last_pre_activation_out = pre_activation_out

        self.out = relu(out)
        return self.out

    def backward(self, dout):
        """
        dout : (N, F, OH, OW)
        Returns: dx, dW, db
        """
        N, F, OH, OW = dout.shape
        
        # ReLU derivative
        d_relu = relu_prime(self.out) * dout
        dout_flat = d_relu.transpose(0, 2, 3, 1).reshape(-1, F)   # (N*OH*OW, F)

        # ---- gradients w.r.t. weights ----
        dW = (dout_flat.T @ self.x_col).reshape(self.W.shape)
        db = dout_flat.sum(axis=0, keepdims=True).T

        # ---- gradient w.r.t. input ----
        dx_col = dout_flat @ self.W_col                              # (N*OH*OW, C*HH*WW)
        dx = col2im(dx_col, self.shape_info,
                    self.W.shape[2], self.W.shape[3])                # (N, C, H, W)

        self.grads = {'dW': dW, 'db': db}
        return dx

    def parameters(self):
        return [self.W, self.b]

    def grads(self):
        return self.grads


# ----------------------------------------------------------------------
#  MaxPool2D
# ----------------------------------------------------------------------
class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool = pool_size
        self.stride = stride
        self.last_x = None
        self.mask = None          # stores indices of max elements

    def forward(self, x):
        N, C, H, W = x.shape

        OH = H // self.pool
        OW = W // self.pool

        x_reshaped = x.reshape(N, C, OH, self.pool, OW, self.pool)
        x_transposed = x_reshaped.transpose(0, 1, 2, 4, 3, 5)
        x_flattened = x_transposed.reshape(N, C, OH, OW, self.pool * self.pool)

        out = np.max(x_flattened, axis=4)
        self.last_x_shape = x.shape

        # Create mask for backward pass
        self.mask = (x_flattened == out[..., np.newaxis])
        return out

    def backward(self, dout):
        """
        dout : (N, C, OH, OW)
        """
        N, C, OH, OW = dout.shape
        dx = np.zeros(self.last_x_shape)

        for n in range(N):
            for c in range(C):
                for i in range(OH):
                    for j in range(OW):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        # Reshape dout to match the flattened patch shape
                        d_patch = np.zeros((self.pool * self.pool,))
                        d_patch[self.mask[n, c, i, j]] = dout[n, c, i, j]
                        dx[n, c,
                           h_start:h_start+self.pool,
                           w_start:w_start+self.pool] += d_patch.reshape(self.pool, self.pool)
        return dx

    def parameters(self):
        return []

    def grads(self):
        return {}


# ----------------------------------------------------------------------
#  Flatten
# ----------------------------------------------------------------------
class Flatten:
    def __init__(self):
        self.input_shape = None

    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.input_shape)

    def parameters(self):
        return []

    def grads(self):
        return {}