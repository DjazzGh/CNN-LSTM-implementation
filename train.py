import numpy as np
from utils.data_loader import MNISTLoader
from utils.optimizers import SGD
from utils.losses import cross_entropy, cross_entropy_prime
from models.model import CNNLSTM

def one_hot(labels, n_classes=10):
    y = np.zeros((len(labels), n_classes))
    y[np.arange(len(labels)), labels] = 1
    return y

def train(model, loader, optimizer, epochs=10):
    for epoch in range(epochs):
        total_loss = 0
        for Xb, yb in loader:
            logits = model.forward(Xb)                # (B,10)
            loss = cross_entropy(logits, one_hot(yb))
            total_loss += loss * len(Xb)

            grad = cross_entropy_prime(logits, one_hot(yb))
            model.backward(grad)                     # implement in model
            optimizer.step(model.parameters(), model.grads())
        print(f"Epoch {epoch+1} loss: {total_loss/len(loader.dataset):.4f}")