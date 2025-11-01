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