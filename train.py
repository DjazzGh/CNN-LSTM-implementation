import numpy as np
from utils.data_loader import MNISTLoader
from utils.optimizers import SGD
from utils.losses import cross_entropy, cross_entropy_prime
from models.model import CNNLSTM


def one_hot(labels, num_classes):
    """Converts a vector of class labels into a one-hot encoded matrix.

    Args:
        labels (numpy.ndarray): A 1D array of integer class labels.
        num_classes (int): The total number of unique classes.

    Returns:
        numpy.ndarray: A 2D array where each row is a one-hot encoded vector
                       corresponding to the input label.
    """
    num_labels = len(labels)
    one_hot_labels = np.zeros((num_labels, num_classes))
    one_hot_labels[np.arange(num_labels), labels] = 1
    return one_hot_labels


def train(model, data_loader, optimizer, num_epochs):
    """Trains the given model using the provided data loader and optimizer.

    Args:
        model: The neural network model to be trained.
        data_loader: An instance of MNISTLoader to provide training data batches.
        optimizer: An instance of an optimizer (e.g., SGD) to update model parameters.
        num_epochs (int): The number of training epochs.
    """
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = data_loader.get_num_train_batches()

        for i in range(num_batches):
            images, labels = data_loader.get_train_batch()

            # Normalize images to [0, 1] and reshape for CNN-LSTM input
            images = images.astype(np.float32) / 255.0
            # Reshape for CNN input (batch, height, width, channels)
            images = images.reshape(-1, 28, 28, 1)

            # One-hot encode labels
            labels_one_hot = one_hot(labels, num_classes=10)

            # Forward pass
            logits = model.forward(images)

            # Calculate loss
            loss = cross_entropy(labels_one_hot, logits)
            total_loss += loss

            # Backward pass
            d_logits = cross_entropy_prime(labels_one_hot, logits)
            model.backward(d_logits)

            # Update model parameters
            optimizer.step(model.parameters(), model.grads())

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")