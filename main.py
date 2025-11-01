# main.py
#
# This script orchestrates the training and evaluation of a Convolutional Neural Network (CNN)
# combined with a Long Short-Term Memory (LSTM) network on the MNIST dataset.
# It defines hyperparameters, loads data, initializes the model, optimizer, and
# then runs the training and evaluation loops.

import numpy as np
import os
import sys

# Add the project root to the Python path to allow importing modules from models and utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.model import CNNLSTM
from utils.data_loader import MNISTLoader
from utils.optimizers import SGD
from train import train
from test import evaluate

def main():
    # Define hyperparameters for the model and training process
    num_classes = 10  # Number of output classes for MNIST (digits 0-9)
    input_shape = (1, 28, 28)  # Shape of the input images (channels, height, width)
    
    # CNN parameters
    cnn_filters = [32, 64]  # Number of filters for each convolutional layer
    cnn_kernel_sizes = [3, 3]  # Kernel size for each convolutional layer
    cnn_strides = [1, 1]  # Stride for each convolutional layer
    cnn_paddings = [1, 1]  # Padding for each convolutional layer
    pool_sizes = [2, 2]  # Pool size for each max-pooling layer

    # LSTM parameters
    lstm_hidden_size = 128  # Number of features in the hidden state of the LSTM
    lstm_sequence_length = 7 * 7  # The sequence length for the LSTM, derived from CNN output
    
    # Training parameters
    learning_rate = 0.01  # Learning rate for the optimizer
    momentum = 0.9  # Momentum for the SGD optimizer
    epochs = 10  # Number of training epochs
    batch_size = 64  # Number of samples per batch
    
    # Initialize data loader
    # Downloads and loads the MNIST dataset, providing data in batches
    data_loader = MNISTLoader(batch_size=batch_size)
    
    # Initialize the CNN-LSTM model
    # The model integrates CNN layers for feature extraction and LSTM for sequence processing
    model = CNNLSTM(
        input_shape=input_shape,
        num_classes=num_classes,
        cnn_filters=cnn_filters,
        cnn_kernel_sizes=cnn_kernel_sizes,
        cnn_strides=cnn_strides,
        cnn_paddings=cnn_paddings,
        pool_sizes=pool_sizes,
        lstm_hidden_size=lstm_hidden_size,
        lstm_sequence_length=lstm_sequence_length
    )
    
    # Initialize the optimizer
    # Stochastic Gradient Descent with momentum is used to update model parameters
    optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
    
    # Training loop
    print("Starting training...")
    for epoch in range(epochs):
        # Train the model for one epoch
        # This involves forward pass, loss calculation, backward pass, and parameter updates
        train_loss, train_accuracy = train(model, data_loader, optimizer)
        
        # Evaluate the model on the test set after each epoch
        # This measures the model's performance on unseen data
        test_accuracy = evaluate(model, data_loader)
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
    
    print("Training finished.")

if __name__ == "__main__":
    main()