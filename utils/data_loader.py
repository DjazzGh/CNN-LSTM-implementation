# utils/data_loader.py
import numpy as np
import os
import gzip
import urllib.request

MNIST_URL = "https://raw.githubusercontent.com/fgnt/mnist/master/"

def download_mnist():
    if not os.path.exists("mnist"):
        os.makedirs("mnist")
    files = ["train-images-idx3-ubyte.gz?raw=true", "train-labels-idx1-ubyte.gz?raw=true",
             "t10k-images-idx3-ubyte.gz?raw=true", "t10k-labels-idx1-ubyte.gz?raw=true"]
    for f in files:
        if not os.path.exists(f"mnist/{f.split('?')[0]}"):
            print(f"Downloading {f.split('?')[0]}...")
            urllib.request.urlretrieve(MNIST_URL + f, f"mnist/{f.split('?')[0]}")

def load_mnist():
    download_mnist()
    def _load_images(path):
        with gzip.open(path, 'rb') as f:
            magic, num, rows, cols = np.frombuffer(f.read(16), '>i4')
            return np.frombuffer(f.read(), 'u1').reshape(num, rows, cols)
    def _load_labels(path):
        with gzip.open(path, 'rb') as f:
            magic, num = np.frombuffer(f.read(8), '>i4')
            return np.frombuffer(f.read(), 'u1')

    X_train = _load_images("mnist/train-images-idx3-ubyte.gz")[..., np.newaxis]  # (N,28,28,1)
    y_train = _load_labels("mnist/train-labels-idx1-ubyte.gz")
    X_test  = _load_images("mnist/t10k-images-idx3-ubyte.gz")[..., np.newaxis]
    y_test  = _load_labels("mnist/t10k-labels-idx1-ubyte.gz")

    return (X_train, y_train), (X_test, y_test)

class MNISTLoader:
    def __init__(self, X, y, batch_size=64, shuffle=True):
        self.X = X.astype(np.float32) / 255.0
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        idx = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(idx)
        for i in range(0, len(self.X), self.batch_size):
            batch_idx = idx[i:i+self.batch_size]
            yield self.X[batch_idx], self.y[batch_idx]