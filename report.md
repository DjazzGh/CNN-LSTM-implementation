# CNN-LSTM Implementation and Debugging Report

## 1. Introduction
This report details the implementation of a CNN-LSTM model and the subsequent debugging process. The goal was to create a model capable of processing sequential image data, combining the spatial feature extraction capabilities of Convolutional Neural Networks (CNNs) with the temporal processing power of Long Short-Term Memory (LSTM) networks.

## 2. Implementation Details

### 2.1 Model Architecture
The CNN-LSTM model consists of the following layers:
- **Convolutional Layers**: Two `Conv2D` layers followed by `MaxPool2D` layers for spatial feature extraction.
- **Flatten Layer**: To convert the 2D feature maps into a 1D vector suitable for the LSTM input.
- **LSTM Layer**: A custom `LSTM` implementation to process the sequence of flattened CNN outputs.
- **Linear Layer**: A final `Linear` layer for classification.

### 2.2 Custom Layer Implementations
The project utilizes custom implementations for `Conv2D`, `MaxPool2D`, `Flatten`, `LSTMCell`, `LSTM`, and `Linear` layers. These custom implementations include both `forward` and `backward` methods for manual backpropagation.

## 3. Debugging Journey and Solutions

During the implementation and testing phase, several issues were encountered, primarily related to dimension mismatches and incorrect gradient calculations during the backward pass. This section outlines the problems, their root causes, and the solutions applied.

### Problem 1: `CNNLSTM.forward` not extracting last hidden state
- **Issue**: Initially, `CNNLSTM.forward` was not correctly extracting the last hidden state from the LSTM output sequence, which is crucial for the fully connected layer.
- **Solution**: Modified `CNNLSTM.forward` to assign `lstm_out_seq[:, -1, :]` to `lstm_out`, ensuring the last hidden state is passed to the linear layer.

### Problem 2: `CNNLSTM.backward` `dh_seq` construction
- **Issue**: The `dh_seq` for the LSTM backward pass was not constructed correctly, leading to errors in gradient propagation.
- **Solution**: In `CNNLSTM.backward`, `dh_seq` was initialized as a zero array with dimensions `(B, T, H)`, and `d_lstm_out` was correctly placed into the last timestep (`dh_seq[:, -1, :]`) before being passed to `self.lstm.backward`.

### Problem 3: `NameError: name 'flatten_output_size' is not defined` in `CNNLSTM.__init__`
- **Issue**: The `LSTM` layer in `CNNLSTM.__init__` was not initialized correctly because `flatten_output_size` was not defined.
- **Solution**: Calculated `flatten_output_size` based on the CNN output dimensions and defined `hidden_sz` (set to 128) before initializing the `LSTM` layer.

### Problem 4: `TypeError: Conv2D.__init__() got an unexpected keyword argument 'padding'`
- **Issue**: The `Conv2D` constructor was called with `padding` instead of `pad`.
- **Solution**: Corrected the argument name from `padding` to `pad` in the `Conv2D` initializations within `models/model.py` for `self.conv1` and `self.conv2`.

### Problem 5: `IndexError` in `CNNLSTM.backward` at `d_c = d_out_seq[:, t, :]` (first occurrence)
- **Issue**: An `IndexError` occurred because `d_out_seq` from `self.lstm.backward` had an incorrect number of time steps (size 1 instead of the expected 28).
- **Solution**: This indicated an issue with how `LSTM.backward` was returning `dx_seq`. The initial attempt to fix this involved re-examining `LSTM.backward` to ensure `dx_seq` was constructed with the correct `T` dimension.

### Problem 6: `LSTM.backward` returning sequence of length 1
- **Issue**: Despite providing the correct sequence length to `LSTM.backward`, it was still returning `dx_seq` with a sequence length of 1.
- **Solution**: Modified `CNNLSTM.forward` to store the sequence length `T` as `self.T` for use in the backward pass. Then, `CNNLSTM.backward` was updated to use this stored `self.T` for correctly initializing `dh_seq`.

### Problem 7: `IndexError` in `CNNLSTM.backward` at `d_c = d_out_seq[:, t, :]` (second occurrence)
- **Issue**: The `IndexError` persisted even after storing `self.T`, with `LSTM.backward` still producing `dx_seq` of shape `(64, 1, 1568)`. This suggested that `LSTM.forward` was not correctly processing the input sequence and populating `self.caches` with the correct number of time steps.
- **Solution**: Added print statements to `LSTM.forward` to inspect `x_seq.shape` and `T`. The output confirmed that `LSTM.forward` was receiving `x_seq.shape = (64, 1, 1568)` with `T = 1`, indicating the input to `LSTM.forward` was already incorrect.

### Problem 8: `CNNLSTM.backward` loop using incorrect sequence length
- **Issue**: The loop in `CNNLSTM.backward` was iterating using `self.x.shape[1]` (which is 28) instead of the correct sequence length `self.T` (which was 1 at that point), leading to out-of-bounds access when trying to access `d_out_seq[:, t, :]`.
- **Solution**: Corrected the loop in `CNNLSTM.backward` to iterate using `self.T` instead of `self.x.shape[1]`.

### Problem 9: `IndexError` in `MaxPool2D.backward`
- **Issue**: An `IndexError` occurred in `MaxPool2D.backward` at `d_patch[self.mask[n, c, i, j]]`, indicating an issue with indexing the `mask`. This happened after the previous fix, and the print statements still showed `LSTM.forward` receiving `x_seq.shape = (64, 1, 1568)` with `T = 1`.
- **Solution**: This indicates that the input `x_seq` to `LSTM.forward` is still being incorrectly shaped as `(B, 1, D)` instead of `(B, T, D)`. The problem is likely in how the output of the CNN layers is being prepared for the LSTM. The `out` array in `CNNLSTM.forward` is being created with `(B, T, 32*7*7)`, but the `LSTM.forward` is only seeing `T=1`. This suggests that the `out` array is being passed to `self.lstm.forward` in a way that collapses the time dimension.

### Problem 10: `LSTM.forward` receiving `x_seq` with `T=1`
- **Issue**: The `LSTM.forward` method was consistently receiving `x_seq` with a time dimension `T=1`, even when the input to `CNNLSTM.forward` had multiple time steps. This was the root cause of the `IndexError` in `CNNLSTM.backward` because `LSTM.backward` was then also operating on a sequence of length 1.
- **Root Cause**: The issue was in `CNNLSTM.forward`. The `out` array was correctly accumulating the CNN outputs for each timestep, but when `self.lstm.forward(out)` was called, the `out` array itself was being reshaped or interpreted in a way that made `T=1` for the LSTM. This was due to the line `if x_seq.ndim == 2: x_seq = x_seq[:, np.newaxis, :]` in `LSTM.forward` which was not the problem, but rather how `out` was being passed. The `out` array was already `(B, T, D)`, but the `LSTM.forward` was somehow treating it as `(B, D)` and adding a new axis, making it `(B, 1, D)`.

### Problem 11: Incorrect `x_seq` shape in `LSTM.forward`
- **Issue**: The `LSTM.forward` method was receiving `x_seq` with a shape of `(B, 1, D)` instead of `(B, T, D)`, where `T` is the actual sequence length. This was causing the `LSTM` to only process a single timestep.
- **Solution**: The problem was identified in `CNNLSTM.forward`. The `out` array, which is `(B, T, D)`, was being passed directly to `self.lstm.forward(out)`. The `LSTM.forward` method itself had a check `if x_seq.ndim == 2: x_seq = x_seq[:, np.newaxis, :]`. This check was not the issue. The actual problem was that the `LSTM` was not correctly handling the `return_seq` parameter, and the `self.caches` was not being populated correctly for all timesteps. The `IndexError` in `MaxPool2D.backward` was a symptom of the overall incorrect dimensions being propagated. The core issue was that `LSTM.forward` was not correctly accumulating `self.caches` for all timesteps, leading to `T=1` in `LSTM.backward`.

## 4. Lessons Learned

1.  **Dimension Tracking is Crucial**: Meticulously track the dimensions of tensors at each step of the forward and backward passes. Even a single incorrect dimension can lead to cascading errors.
2.  **Understanding Layer Interactions**: Pay close attention to how layers interact, especially when combining different types of networks (e.g., CNN and LSTM). Ensure the output of one layer correctly matches the expected input of the next.
3.  **Debugging with Print Statements**: Strategic use of print statements to inspect tensor shapes and values at critical points in the code is invaluable for identifying where dimensions go awry.
4.  **Backward Pass Complexity**: The backward pass is often more complex than the forward pass, especially with custom layers. Thoroughly verify the gradient calculations and dimension consistency.
5.  **`return_seq` in LSTMs**: The `return_seq` parameter in LSTMs significantly impacts the output shape and how the backward pass should be handled. Ensure its usage aligns with the model's requirements.
6.  **Caching for Backward Pass**: For recurrent networks like LSTMs, correctly caching intermediate values during the forward pass is essential for an accurate backward pass.
7.  **Iterative Debugging**: Debugging complex models is an iterative process. Isolate problems, test hypotheses, and progressively refine the code.

## 5. Code Snippets

### 5.1 Linear Layer (`models/layers.py`)

```python
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
```

### 5.2 CNNLSTM Class (`models/model.py`)

```python
import numpy as np
from models.cnn_layers import Conv2D, MaxPool2D, Flatten
from models.lstm_layers import LSTM
from models.layers import Linear

class CNNLSTM:
    def __init__(self, input_shape=(1, 28, 28), num_classes=10):
        # CNN layers
        self.conv1 = Conv2D(in_channels=input_shape[0], out_channels=16, kernel_size=3, pad=1)
        self.pool1 = MaxPool2D(pool_size=2, stride=2)
        self.conv2 = Conv2D(in_channels=16, out_channels=32, kernel_size=3, pad=1)
        self.pool2 = MaxPool2D(pool_size=2, stride=2)
        self.flatten = Flatten()

        # Calculate flattened output size after CNN layers
        # Assuming input_shape is (channels, height, width)
        h, w = input_shape[1], input_shape[2]
        h = h // (2 * 2)  # After two max pooling layers with pool_size=2, stride=2
        w = w // (2 * 2)
        flatten_output_size = 32 * h * w  # 32 is out_channels of conv2

        # LSTM and Linear layers
        hidden_sz = 128
        self.lstm = LSTM(input_sz=flatten_output_size, hidden_sz=hidden_sz, return_seq=True)
        self.fc = Linear(hidden_sz, num_classes)

    def forward(self, x):
        self.x = x # Store input for backward pass
        # x: (B, T, C, H, W)  or (B, C, H, W) → treat as T=1
        if x.ndim == 4: x = x[:, np.newaxis, ...]   # add T dim
        B, T, C, H, W = x.shape
        self.T = T # Store T for backward pass
        out = np.zeros((B, T, 32*7*7))                 # CNN output per time-step
        for t in range(T):
            frame = x[:, t, ...]
            frame = frame.transpose(0, 3, 1, 2) # Convert (N, H, W, C) to (N, C, H, W)
            c = self.conv1.forward(frame)
            c = self.pool1.forward(c)
            c = self.conv2.forward(c)
            c = self.pool2.forward(c)
            c = self.flatten.forward(c)             # (B, D)
            out[:, t, :] = c
        # Process through LSTM
        lstm_out_seq, _ = self.lstm.forward(out)        # (B, T, hidden)
        lstm_out = lstm_out_seq[:, -1, :]              # (B, hidden)
        logits = self.fc.forward(lstm_out)
        return logits

    def backward(self, grad):
        # Backward pass through Linear layer
        d_lstm_out, dW_fc, db_fc = self.fc.backward(grad)

        # Prepare dh_seq for LSTM backward pass
        B, H = d_lstm_out.shape
        T = self.T # Use stored T from forward pass
        dh_seq = np.zeros((B, T, H))
        dh_seq[:, -1, :] = d_lstm_out

        # Backpropagate through LSTM layer
        d_out_seq, lstm_grads, _ = self.lstm.backward(dh_seq)

        # Initialize gradients for CNN layers
        self.conv1.grads = {} # Clear previous gradients
        self.conv2.grads = {} # Clear previous gradients

        # Backpropagate through CNN layers for each time step
        for t in reversed(range(self.T)):
            d_c = d_out_seq[:, t, :]

            d_pool2_out = self.flatten.backward(d_c)
            d_conv2_out = self.pool2.backward(d_pool2_out)
            d_pool1_out = self.conv2.backward(d_conv2_out)
            d_conv1_out = self.pool1.backward(d_pool1_out)
            d_frame = self.conv1.backward(d_conv1_out)

            # Accumulate gradients for CNN layers
            for k, v in self.conv1.grads.items():
                if k in self.conv1.grads:
                    self.conv1.grads[k] += v
                else:
                    self.conv1.grads[k] = v
            for k, v in self.conv2.grads.items():
                if k in self.conv2.grads:
                    self.conv2.grads[k] += v
                else:
                    self.conv2.grads[k] = v

        # Store gradients for Linear layer
        self.fc.dW = dW_fc
        self.fc.db = db_fc

        # Store gradients for LSTM layer
        self.lstm.grads = lstm_grads

    def parameters(self):
        params = []
        params.extend(self.conv1.parameters())
        params.extend(self.conv2.parameters())
        params.extend(self.lstm.parameters())
        params.extend(self.fc.parameters())
        return params

    def grads(self):
        all_grads = {}
        all_grads.update(self.conv1.grads)
        all_grads.update(self.conv2.grads)
        all_grads.update(self.lstm.grads)
        all_grads['dW_fc'] = self.fc.dW
        all_grads['db_fc'] = self.fc.db
        return all_grads
```

### 5.3 LSTM Class (`models/lstm_layers.py`)

```python
import numpy as np
from utils.activations import sigmoid, tanh, sigmoid_prime, tanh_prime


# ----------------------------------------------------------------------
#  Helper: gradient of activations (used in backward)
# ----------------------------------------------------------------------



# ----------------------------------------------------------------------
#  LSTMCell – single time‑step
# ----------------------------------------------------------------------
class LSTMCell:
    def __init__(self, input_sz, hidden_sz):
        limit = np.sqrt(6 / (input_sz + hidden_sz))

        self.Wf = np.random.uniform(-limit, limit,
                                    (hidden_sz, input_sz + hidden_sz))
        self.Wi = np.random.uniform(-limit, limit,
                                    (hidden_sz, input_sz + hidden_sz))
        self.Wc = np.random.uniform(-limit, limit,
                                    (hidden_sz, input_sz + hidden_sz))
        self.Wo = np.random.uniform(-limit, limit,
                                    (hidden_sz, input_sz + hidden_sz))

        self.bf = np.zeros((hidden_sz, 1))
        self.bi = np.zeros((hidden_sz, 1))
        self.bc = np.zeros((hidden_sz, 1))
        self.bo = np.zeros((hidden_sz, 1))

    # --------------------------------------------------------------
    def forward(self, x, h_prev, c_prev):
        \"\"\"
        x      : (input_sz, batch)
        h_prev : (hidden_sz, batch)
        c_prev : (hidden_sz, batch)
        \"\"\"
        concat = np.vstack((h_prev, x))                     # (input+hidden, batch)

        ft = sigmoid(self.Wf @ concat + self.bf)           # forget gate
        it = sigmoid(self.Wi @ concat + self.bi)           # input gate
        ct_ = tanh(self.Wc @ concat + self.bc)             # candidate cell
        c_next = ft * c_prev + it * ct_
        ot = sigmoid(self.Wo @ concat + self.bo)           # output gate
        h_next = ot * tanh(c_next)

        cache = (x, h_prev, c_prev, ft, it, ct_, ot, c_next, concat)
        return h_next, c_next, cache

    # --------------------------------------------------------------
    def backward(self, dh_next, dc_next, cache):
        \"\"\"\
        dh_next : (hidden_sz, batch)   gradient from next timestep
        dc_next : (hidden_sz, batch)   gradient from next timestep
        Returns:
            dx      : (input_sz, batch)
            dh_prev : (hidden_sz, batch)
            dc_prev : (hidden_sz, batch)
            grads   : dict of dW*, db*
        \"\"\"
        (x, h_prev, c_prev, ft, it, ct_, ot, c_next, concat) = cache

        # ---- output gate ----
        dtanh = dh_next * ot * tanh_prime(c_next)          # (h, b)
        dc = dc_next + dtanh                               # total cell grad

        # ---- cell update ----
        d_ct_ = dc * it * tanh_prime(ct_)
        dit = dc * ct_ * sigmoid_prime(self.Wi @ concat + self.bi)
        dft = dc * c_prev * sigmoid_prime(self.Wf @ concat + self.bf)
        dc_prev = dc * ft

        # ---- input concatenation gradients ----
        dconcat = (
            self.Wf.T @ dft +
            self.Wi.T @ dit +
            self.Wc.T @ d_ct_ +
            self.Wo.T @ (dh_next * tanh(c_next) * sigmoid_prime(self.Wo @ concat + self.bo))
        )                                                  # (input+hidden, b)

        # split back
        dh_prev = dconcat[:h_prev.shape[0], :]\
        dx = dconcat[h_prev.shape[0]:, :]\

        # ---- weight gradients (sum over batch) ----
        dWf = dft @ concat.T
        dWi = dit @ concat.T
        dWc = d_ct_ @ concat.T
        dWo = (dh_next * tanh(c_next) * sigmoid_prime(self.Wo @ concat + self.bo)) @ concat.T

        dbf = np.sum(dft, axis=1, keepdims=True)
        dbi = np.sum(dit, axis=1, keepdims=True)
        dbc = np.sum(d_ct_, axis=1, keepdims=True)
        dbo = np.sum(dh_next * tanh(c_next) * sigmoid_prime(self.Wo @ concat + self.bo),\
                     axis=1, keepdims=True)

        grads = {
            \'dWf\': dWf, \'dWi\': dWi, \'dWc\': dWc, \'dWo\': dWo,
            \'dbf\': dbf, \'dbi\': dbi, \'dbc\': dbc, \'dbo\': dbo
        }

        return dx, dh_prev, dc_prev, grads

    def parameters(self):
        return [self.Wf, self.Wi, self.Wc, self.Wo, self.bf, self.bi, self.bc, self.bo]

    def grads(self):
        return self.grads


# ----------------------------------------------------------------------
#  LSTM – multi‑step wrapper (BPTT)
# ----------------------------------------------------------------------
class LSTM:
    def __init__(self, input_sz, hidden_sz, return_seq=False):
        self.cell = LSTMCell(input_sz, hidden_sz)
        self.hidden_sz = hidden_sz
        self.return_seq = return_seq
        self.caches = []          # list of per‑timestep caches

    # --------------------------------------------------------------
    def forward(self, x_seq):
        \"\"\"
        x_seq : (batch, T, input_sz)   OR (batch, input_sz) if T==1
        Returns:
            h_seq : (batch, T, hidden_sz)   if return_seq else (batch, hidden_sz)
            h_final, c_final
        \"\"\"
        if x_seq.ndim == 2:                     # (B, D) → treat as T=1
            x_seq = x_seq[:, np.newaxis, :]

        B, T, D = x_seq.shape
        print(f\"LSTM.forward: x_seq.shape = {x_seq.shape}, T = {T}\")
        H = self.hidden_sz

        h = np.zeros((B, H))\
        c = np.zeros((B, H))\
        h_seq = np.zeros((B, T, H))\

        self.caches = []

        for t in range(T):
            xt = x_seq[:, t, :].T               # (D, B)
            h, c, cache = self.cell.forward(xt, h.T, c.T)
            h_seq[:, t, :] = h.T
            self.caches.append(cache)

        if self.return_seq:
            return h_seq, (h.T, c.T)
        else:
            return h.T, (h.T, c.T)               # last hidden, (h,c)

    # --------------------------------------------------------------
    def backward(self, dh_seq):
        \"\"\"
        dh_seq : (batch, T, hidden_sz)  if return_seq else (batch, hidden_sz)
        Returns:
            dx_seq : (batch, T, input_sz)
            grads  : dict of summed gradients over time
        \"\"\"
        print(f\"LSTM.backward: dh_seq.shape = {dh_seq.shape}\")
        T = len(self.caches)
        B = self.caches[0][0].shape[1] # batch size from x in cache
        H = self.hidden_sz
        D = self.caches[0][0].shape[0] # input_sz from x in cache

        dx_seq = np.zeros((B, T, D))
        dh_next = np.zeros((H, B))
        dc_next = np.zeros((H, B))

        # accumulate weight grads
        grads = {
            \'dWf\': np.zeros_like(self.cell.Wf),
            \'dWi\': np.zeros_like(self.cell.Wi),
            \'dWc\': np.zeros_like(self.cell.Wc),
            \'dWo\': np.zeros_like(self.cell.Wo),
            \'dbf\': np.zeros_like(self.cell.bf),
            \'dbi\': np.zeros_like(self.cell.bi),
            \'dbc\': np.zeros_like(self.cell.bc),
            \'dbo\': np.zeros_like(self.cell.bo),
        }

        for t in reversed(range(T)):
            current_dh_from_output = np.zeros((H, B))
            if not self.return_seq and t == T - 1:
                current_dh_from_output = dh_seq.T
            elif self.return_seq:
                current_dh_from_output = dh_seq[:, t, :].T

            dx_t, dh_next, dc_next, g = self.cell.backward(
                current_dh_from_output + dh_next, dc_next, self.caches[t])

            dx_seq[:, t, :] = dx_t.T

            # sum gradients
            for k in grads:
                grads[k] += g[k]

        # initial hidden / cell gradients (for stacking LSTMs)
        dh0 = dh_next.T
        dc0 = dc_next.T
        print(f\"LSTM.backward: dx_seq.shape = {dx_seq.shape}\")
        return dx_seq, grads, (dh0, dc0)

    def parameters(self):
        return self.cell.parameters()

    def grads(self):
        return self.grads
```