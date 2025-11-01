import numpy as np
import gzip
import os

def download_mnist(path):
    """Downloads the MNIST dataset if it's not already present.

    Args:
        path (str): The directory where the MNIST dataset should be stored.
    """
    url = 'http://yann.lecun.com/exdb/mnist/'
    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]

    for file in files:
        filepath = os.path.join(path, file)
        if not os.path.exists(filepath):
            print(f"Downloading {file}...")
            # In a real scenario, you would use a library like `requests` to download the file.
            # For this example, we'll assume the files are already present or manually downloaded.
            # Example: requests.get(url + file, stream=True)
            print(f"Please ensure {file} is available at {path}")

def load_mnist(path, kind='train'):
    """Loads MNIST image and label data from gzipped files.

    Args:
        path (str): The directory where the MNIST dataset files are located.
        kind (str, optional): Specifies whether to load 'train' or 't10k' (test) data.
                              Defaults to 'train'.

    Returns:
        tuple: A tuple containing:
            - images (numpy.ndarray): An array of MNIST images.
            - labels (numpy.ndarray): An array of MNIST labels.
    """
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels


class MNISTLoader:
    """A data loader for the MNIST dataset, providing batching and shuffling capabilities.
    """
    def __init__(self, path, batch_size, shuffle=True):
        """Initializes the MNIST data loader.

        Args:
            path (str): The directory where the MNIST dataset files are located.
            batch_size (int): The number of samples per batch.
            shuffle (bool, optional): Whether to shuffle the dataset after each epoch. Defaults to True.
        """
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Load training and test data
        self.train_images, self.train_labels = load_mnist(path, kind='train')
        self.test_images, self.test_labels = load_mnist(path, kind='t10k')

        self.num_train_samples = len(self.train_images)
        self.num_test_samples = len(self.test_images)

        self.train_idx = np.arange(self.num_train_samples)
        self.test_idx = np.arange(self.num_test_samples)

        self.current_train_batch = 0
        self.current_test_batch = 0

        if self.shuffle:
            np.random.shuffle(self.train_idx)

    def get_train_batch(self):
        """Generates a batch of training images and labels.

        Returns:
            tuple: A tuple containing:
                - batch_images (numpy.ndarray): A batch of training images.
                - batch_labels (numpy.ndarray): A batch of training labels.
        """
        start = self.current_train_batch * self.batch_size
        end = (self.current_train_batch + 1) * self.batch_size

        if end > self.num_train_samples:
            if self.shuffle:
                np.random.shuffle(self.train_idx)
            self.current_train_batch = 0
            start = 0
            end = self.batch_size

        batch_idx = self.train_idx[start:end]
        batch_images = self.train_images[batch_idx]
        batch_labels = self.train_labels[batch_idx]

        self.current_train_batch += 1
        return batch_images, batch_labels

    def get_test_batch(self):
        """Generates a batch of test images and labels.

        Returns:
            tuple: A tuple containing:
                - batch_images (numpy.ndarray): A batch of test images.
                - batch_labels (numpy.ndarray): A batch of test labels.
        """
        start = self.current_test_batch * self.batch_size
        end = (self.current_test_batch + 1) * self.batch_size

        if end > self.num_test_samples:
            self.current_test_batch = 0
            start = 0
            end = self.batch_size

        batch_idx = self.test_idx[start:end]
        batch_images = self.test_images[batch_idx]
        batch_labels = self.test_labels[batch_idx]

        self.current_test_batch += 1
        return batch_images, batch_labels

    def get_num_train_batches(self):
        """Returns the total number of training batches.

        Returns:
            int: The number of training batches.
        """
        return (self.num_train_samples + self.batch_size - 1) // self.batch_size

    def get_num_test_batches(self):
        """Returns the total number of test batches.

        Returns:
            int: The number of test batches.
        """
        return (self.num_test_samples + self.batch_size - 1) // self.batch_size