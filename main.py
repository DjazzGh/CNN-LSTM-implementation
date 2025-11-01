import numpy as np
from utils.data_loader import MNISTLoader, load_mnist, download_mnist
from utils.optimizers import SGD
from models.model import CNNLSTM
from train import train
from test import evaluate

# Hyperparameters
epochs = 10
batch_size = 64
learning_rate = 0.01

# Data loading
download_mnist()
(X_train, y_train), (X_test, y_test) = load_mnist()
train_loader = MNISTLoader(X_train, y_train, batch_size=batch_size)
test_loader = MNISTLoader(X_test, y_test, batch_size=batch_size)

# Model, optimizer, and loss
model = CNNLSTM()
optimizer = SGD(lr=learning_rate)

# Training
train(model, train_loader, optimizer, epochs=epochs)

# Evaluation
accuracy = evaluate(model, test_loader)
print(f"Test Accuracy: {accuracy:.4f}")