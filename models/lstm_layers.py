# models/lstm_layers.py
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
        """
        x      : (input_sz, batch)
        h_prev : (hidden_sz, batch)
        c_prev : (hidden_sz, batch)
        """
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
        """
        dh_next : (hidden_sz, batch)   gradient from next timestep
        dc_next : (hidden_sz, batch)   gradient from next timestep
        Returns:
            dx      : (input_sz, batch)
            dh_prev : (hidden_sz, batch)
            dc_prev : (hidden_sz, batch)
            grads   : dict of dW*, db*
        """
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
        dh_prev = dconcat[:h_prev.shape[0], :]
        dx = dconcat[h_prev.shape[0]:, :]

        # ---- weight gradients (sum over batch) ----
        dWf = dft @ concat.T
        dWi = dit @ concat.T
        dWc = d_ct_ @ concat.T
        dWo = (dh_next * tanh(c_next) * sigmoid_prime(self.Wo @ concat + self.bo)) @ concat.T

        dbf = np.sum(dft, axis=1, keepdims=True)
        dbi = np.sum(dit, axis=1, keepdims=True)
        dbc = np.sum(d_ct_, axis=1, keepdims=True)
        dbo = np.sum(dh_next * tanh(c_next) * sigmoid_prime(self.Wo @ concat + self.bo),
                     axis=1, keepdims=True)

        grads = {
            'dWf': dWf, 'dWi': dWi, 'dWc': dWc, 'dWo': dWo,
            'dbf': dbf, 'dbi': dbi, 'dbc': dbc, 'dbo': dbo
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
        """
        x_seq : (batch, T, input_sz)   OR (batch, input_sz) if T==1
        Returns:
            h_seq : (batch, T, hidden_sz)   if return_seq else (batch, hidden_sz)
            h_final, c_final
        """
        if x_seq.ndim == 2:                     # (B, D) → treat as T=1
            x_seq = x_seq[:, np.newaxis, :]

        B, T, D = x_seq.shape
        print(f"LSTM.forward: x_seq.shape = {x_seq.shape}, T = {T}")
        H = self.hidden_sz

        h = np.zeros((B, H))
        c = np.zeros((B, H))
        h_seq = np.zeros((B, T, H))

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
        """
        dh_seq : (batch, T, hidden_sz)  if return_seq else (batch, hidden_sz)
        Returns:
            dx_seq : (batch, T, input_sz)
            grads  : dict of summed gradients over time
        """
        print(f"LSTM.backward: dh_seq.shape = {dh_seq.shape}")
        T = len(self.caches)
        B = self.caches[0][0].shape[1] # batch size from x in cache
        H = self.hidden_sz
        D = self.caches[0][0].shape[0] # input_sz from x in cache

        dx_seq = np.zeros((B, T, D))
        dh_next = np.zeros((H, B))
        dc_next = np.zeros((H, B))

        # accumulate weight grads
        grads = {
            'dWf': np.zeros_like(self.cell.Wf),
            'dWi': np.zeros_like(self.cell.Wi),
            'dWc': np.zeros_like(self.cell.Wc),
            'dWo': np.zeros_like(self.cell.Wo),
            'dbf': np.zeros_like(self.cell.bf),
            'dbi': np.zeros_like(self.cell.bi),
            'dbc': np.zeros_like(self.cell.bc),
            'dbo': np.zeros_like(self.cell.bo),
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
        print(f"LSTM.backward: dx_seq.shape = {dx_seq.shape}")
        return dx_seq, grads, (dh0, dc0)

    def parameters(self):
        return self.cell.parameters()

    def grads(self):
        return self.grads