import numpy as np
from utils.activations import sigmoid, sigmoid_prime, tanh, tanh_prime


class LSTMCell:
    """A single Long Short-Term Memory (LSTM) cell.

    This class implements the forward and backward passes for a single time step
    of an LSTM network. It handles the input, forget, output, and cell gate computations.
    """
    def __init__(self, input_size, hidden_size):
        """Initializes the LSTM cell with weights and biases for all gates.

        Args:
            input_size (int): The number of expected features in the input x.
            hidden_size (int): The number of features in the hidden state h.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights for input, forget, output, and cell gates
        # Weights are concatenated for efficiency: W_xi, W_xf, W_xo, W_xg
        self.W_x = np.random.randn(input_size, 4 * hidden_size) * np.sqrt(2. / (input_size + hidden_size))
        # Weights for recurrent connections: W_hi, W_hf, W_ho, W_hg
        self.W_h = np.random.randn(hidden_size, 4 * hidden_size) * np.sqrt(2. / (input_size + hidden_size))

        # Biases for input, forget, output, and cell gates
        self.b = np.zeros((1, 4 * hidden_size))

        # Gradients
        self.dW_x = None
        self.dW_h = None
        self.db = None

        # Caches for backward pass
        self.x = None
        self.h_prev = None
        self.c_prev = None
        self.i = None
        self.f = None
        self.o = None
        self.g = None
        self.c_next = None

    def forward(self, x, h_prev, c_prev):
        """Performs the forward pass for one time step of the LSTM cell.

        Args:
            x (numpy.ndarray): Input at the current time step (batch_size, input_size).
            h_prev (numpy.ndarray): Previous hidden state (batch_size, hidden_size).
            c_prev (numpy.ndarray): Previous cell state (batch_size, hidden_size).

        Returns:
            tuple: A tuple containing:
                - h_next (numpy.ndarray): Current hidden state (batch_size, hidden_size).
                - c_next (numpy.ndarray): Current cell state (batch_size, hidden_size).
        """
        self.x = x
        self.h_prev = h_prev
        self.c_prev = c_prev

        # Concatenate input and previous hidden state for matrix multiplication
        concat_input = np.concatenate((x, h_prev), axis=1)

        # Compute activations for all four gates (input, forget, output, cell_candidate)
        gates = np.dot(x, self.W_x) + np.dot(h_prev, self.W_h) + self.b

        # Split the gates into individual components
        i_gate = gates[:, :self.hidden_size]  # Input gate
        f_gate = gates[:, self.hidden_size:2 * self.hidden_size]  # Forget gate
        o_gate = gates[:, 2 * self.hidden_size:3 * self.hidden_size]  # Output gate
        g_gate = gates[:, 3 * self.hidden_size:]  # Cell gate (candidate cell state)

        # Apply activation functions
        self.i = sigmoid(i_gate)
        self.f = sigmoid(f_gate)
        self.o = sigmoid(o_gate)
        self.g = tanh(g_gate)

        # Compute the next cell state
        c_next = self.f * c_prev + self.i * self.g
        # Compute the next hidden state
        h_next = self.o * tanh(c_next)

        self.c_next = c_next # Cache for backward pass
        return h_next, c_next

    def backward(self, dh_next, dc_next):
        """Performs the backward pass for one time step of the LSTM cell.

        Computes gradients with respect to input, previous hidden state, previous cell state,
        and all weights and biases.

        Args:
            dh_next (numpy.ndarray): Gradient of the loss with respect to the next hidden state.
            dc_next (numpy.ndarray): Gradient of the loss with respect to the next cell state.

        Returns:
            tuple: A tuple containing:
                - dx (numpy.ndarray): Gradient with respect to the input x.
                - dh_prev (numpy.ndarray): Gradient with respect to the previous hidden state.
                - dc_prev (numpy.ndarray): Gradient with respect to the previous cell state.
        """
        # Gradient of tanh(c_next) in h_next calculation
        dtanh_c_next = tanh_prime(self.c_next) * self.o * dh_next

        # Total gradient for c_next
        dc_next_combined = dc_next + dtanh_c_next

        # Gradients for gates and c_prev
        df_gate = sigmoid_prime(self.f) * self.c_prev * dc_next_combined
        di_gate = sigmoid_prime(self.i) * self.g * dc_next_combined
        dg_gate = tanh_prime(self.g) * self.i * dc_next_combined
        do_gate = sigmoid_prime(self.o) * tanh(self.c_next) * dh_next

        dc_prev = self.f * dc_next_combined

        # Concatenate gate gradients
        d_gates = np.concatenate((di_gate, df_gate, do_gate, dg_gate), axis=1)

        # Gradients for W_x, W_h, b
        self.dW_x = np.dot(self.x.T, d_gates)
        self.dW_h = np.dot(self.h_prev.T, d_gates)
        self.db = np.sum(d_gates, axis=0, keepdims=True)

        # Gradients for x and h_prev
        dx = np.dot(d_gates, self.W_x.T)
        dh_prev = np.dot(d_gates, self.W_h.T)

        return dx, dh_prev, dc_prev


class LSTM:
    """A multi-step Long Short-Term Memory (LSTM) layer.

    This class wraps the LSTMCell to process sequences of inputs, handling
    the forward pass through multiple time steps and backpropagation through time (BPTT).
    """
    def __init__(self, input_size, hidden_size, sequence_length):
        """Initializes the LSTM layer.

        Args:
            input_size (int): The number of expected features in the input x.
            hidden_size (int): The number of features in the hidden state h.
            sequence_length (int): The length of the input sequence.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

        self.lstm_cell = LSTMCell(input_size, hidden_size)

        # Caches for backward pass
        self.h_states = None
        self.c_states = None
        self.x_sequence = None

    def forward(self, x_sequence):
        """Performs the forward pass through the LSTM layer for a sequence of inputs.

        Args:
            x_sequence (numpy.ndarray): Input sequence of shape (batch_size, sequence_length, input_size).

        Returns:
            numpy.ndarray: The hidden state of the last time step (batch_size, hidden_size).
        """
        batch_size = x_sequence.shape[0]

        # Initialize hidden and cell states for the first time step
        h = np.zeros((batch_size, self.hidden_size))
        c = np.zeros((batch_size, self.hidden_size))

        # Store all hidden and cell states for backpropagation through time
        self.h_states = np.zeros((batch_size, self.sequence_length, self.hidden_size))
        self.c_states = np.zeros((batch_size, self.sequence_length, self.hidden_size))
        self.x_sequence = x_sequence # Cache input sequence

        for t in range(self.sequence_length):
            x_t = x_sequence[:, t, :]
            h, c = self.lstm_cell.forward(x_t, h, c)
            self.h_states[:, t, :] = h
            self.c_states[:, t, :] = c

        return h # Return the last hidden state

    def backward(self, dh_last):
        """Performs the backward pass through the LSTM layer using Backpropagation Through Time (BPTT).

        Args:
            dh_last (numpy.ndarray): Gradient of the loss with respect to the last hidden state.

        Returns:
            numpy.ndarray: Gradient with respect to the input sequence.
        """
        batch_size = dh_last.shape[0]

        # Initialize gradients for previous hidden and cell states
        dh_prev = dh_last
        dc_prev = np.zeros_like(dh_last)

        # Initialize gradients for input sequence
        dx_sequence = np.zeros_like(self.x_sequence)

        # Initialize gradients for weights and biases
        self.dW_x = np.zeros_like(self.lstm_cell.W_x)
        self.dW_h = np.zeros_like(self.lstm_cell.W_h)
        self.db = np.zeros_like(self.lstm_cell.b)

        # Backpropagate through time
        for t in reversed(range(self.sequence_length)):
            x_t = self.x_sequence[:, t, :]
            h_t = self.h_states[:, t, :]
            c_t = self.c_states[:, t, :]

            # Get previous hidden and cell states from cache (or zeros for t=0)
            h_prev_t = self.h_states[:, t - 1, :] if t > 0 else np.zeros_like(h_t)
            c_prev_t = self.c_states[:, t - 1, :] if t > 0 else np.zeros_like(c_t)

            # Perform backward pass for the current LSTM cell
            dx_t, dh_prev, dc_prev = self.lstm_cell.backward(dh_prev, dc_prev)

            # Accumulate gradients for weights and biases
            self.dW_x += self.lstm_cell.dW_x
            self.dW_h += self.lstm_cell.dW_h
            self.db += self.lstm_cell.db

            # Store gradient for the input sequence
            dx_sequence[:, t, :] = dx_t

        return dx_sequence

    def parameters(self):
        """Returns the learnable parameters of the LSTM layer.

        Returns:
            dict: A dictionary containing weights and biases for all gates.
        """
        return {
            'Wf': self.lstm_cell.W_x[:, self.hidden_size:2 * self.hidden_size],
            'Wi': self.lstm_cell.W_x[:, :self.hidden_size],
            'Wo': self.lstm_cell.W_x[:, 2 * self.hidden_size:3 * self.hidden_size],
            'Wg': self.lstm_cell.W_x[:, 3 * self.hidden_size:],
            'Uf': self.lstm_cell.W_h[:, self.hidden_size:2 * self.hidden_size],
            'Ui': self.lstm_cell.W_h[:, :self.hidden_size],
            'Uo': self.lstm_cell.W_h[:, 2 * self.hidden_size:3 * self.hidden_size],
            'Ug': self.lstm_cell.W_h[:, 3 * self.hidden_size:],
            'bf': self.lstm_cell.b[:, self.hidden_size:2 * self.hidden_size],
            'bi': self.lstm_cell.b[:, :self.hidden_size],
            'bo': self.lstm_cell.b[:, 2 * self.hidden_size:3 * self.hidden_size],
            'bg': self.lstm_cell.b[:, 3 * self.hidden_size:]
        }

    def grads(self):
        """Returns the gradients of the learnable parameters of the LSTM layer.

        Returns:
            dict: A dictionary containing gradients for weights and biases for all gates.
        """
        return {
            'dWf': self.dW_x[:, self.hidden_size:2 * self.hidden_size],
            'dWi': self.dW_x[:, :self.hidden_size],
            'dWo': self.dW_x[:, 2 * self.hidden_size:3 * self.hidden_size],
            'dWg': self.dW_x[:, 3 * self.hidden_size:],
            'dUf': self.dW_h[:, self.hidden_size:2 * self.hidden_size],
            'dUi': self.dW_h[:, :self.hidden_size],
            'dUo': self.dW_h[:, 2 * self.hidden_size:3 * self.hidden_size],
            'dUg': self.dW_h[:, 3 * self.hidden_size:],
            'dbf': self.db[:, self.hidden_size:2 * self.hidden_size],
            'dbi': self.db[:, :self.hidden_size],
            'dbo': self.db[:, 2 * self.hidden_size:3 * self.hidden_size],
            'dbg': self.db[:, 3 * self.hidden_size:]
        }