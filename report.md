# CNN-LSTM Implementation and Debugging Report

## 1. Project Structure Overview

The project is structured into several key directories:
- `main.py`: The entry point of the application, responsible for setting up hyperparameters, loading data, initializing the model and optimizer, and orchestrating the training and evaluation process.
- `mnist/`: Contains the MNIST dataset files.
- `models/`: Houses the neural network layer implementations and the main CNN-LSTM model.
    - `cnn_layers.py`: Implements 2D convolutional layers, max-pooling, and flattening. It also includes `im2col` and `col2im` utilities for efficient convolution.
    - `lstm_layers.py`: Implements the LSTM cell and the full LSTM layer, handling sequential data processing.
    - `layers.py`: Contains basic neural network layers, specifically the `Linear` (fully connected) layer.
    - `model.py`: Defines the `CNNLSTM` class, which integrates the CNN and LSTM components to form the complete model architecture.
- `myenv/`: A virtual environment for the project dependencies.
- `utils/`: Provides utility functions for activations, data loading, loss calculations, and optimizers.
    - `activations.py`: Implements activation functions like ReLU, Sigmoid, Tanh, and Softmax, along with their derivatives.
    - `data_loader.py`: Handles downloading and loading the MNIST dataset, and provides a `MNISTLoader` class for batching and shuffling data.
    - `losses.py`: Implements the cross-entropy loss function and its derivative.
    - `optimizers.py`: Contains the implementation of optimization algorithms, specifically Stochastic Gradient Descent (SGD).
- `train.py`: Contains the `train` function, which orchestrates the training loop, including forward pass, loss calculation, backward pass, and optimizer step.
- `test.py`: Contains the `evaluate` function, which performs inference on the test set and calculates accuracy.
- `requirements.txt`: Lists the project's Python dependencies.

## 2. Model Architecture (CNNLSTM)

The `CNNLSTM` model, defined in `models/model.py`, combines Convolutional Neural Network (CNN) layers for feature extraction from image data with a Long Short-Term Memory (LSTM) layer for processing sequential information.

### 2.1. CNN Component

The CNN part of the model consists of:
- Two `Conv2D` layers: Each followed by a ReLU activation (implicitly handled within `Conv2D.forward` due to `relu` being applied to `self.out`).
    - `conv1`: 1 input channel, 16 output channels, 3x3 kernel, 1-pixel padding.
    - `conv2`: 16 input channels, 32 output channels, 3x3 kernel, 1-pixel padding.
- Two `MaxPool2D` layers: Each with a 2x2 pool size and a stride of 2.
- A `Flatten` layer: Converts the 2D feature maps into a 1D vector.

The `Conv2D` layer uses `im2col` for efficient convolution operations, converting image patches into columns to perform matrix multiplications. The `MaxPool2D` layer performs downsampling by selecting the maximum value within each pooling window. The `Flatten` layer prepares the output of the CNN for the subsequent LSTM layer.

### 2.2. LSTM Component

The output of the CNN (flattened feature vector for each time step) is fed into an `LSTM` layer.
- `LSTM`: Takes the flattened CNN output as input and processes it sequentially. The `hidden_sz` is set to 128. The `return_seq` parameter is set to `True` during initialization, but the `forward` method extracts the last hidden state for the final classification.

The `LSTMCell` (within `lstm_layers.py`) implements the core LSTM logic, including forget, input, candidate, and output gates, using sigmoid and tanh activation functions.

### 2.3. Output Layer

Finally, a `Linear` (fully connected) layer maps the output of the LSTM to the number of classes (10 for MNIST).
- `Linear`: Takes the `hidden_sz` (128) from the LSTM as input and outputs 10 values (logits) corresponding to the 10 MNIST digits.

### 2.4. Forward Pass

The `forward` method of `CNNLSTM` processes input `x` (which can be a single image or a sequence of images). If a single image, it adds a time dimension. It then iterates through the time steps, applying the CNN layers to each frame. The output of the CNN for each time step is collected and then passed to the LSTM layer. The final output is the logits from the linear layer applied to the last hidden state of the LSTM.

### 2.5. Backward Pass

The `backward` method implements backpropagation through the entire CNN-LSTM network. It starts by backpropagating through the `Linear` layer, then the `LSTM` layer, and finally iterates backward through the time steps to backpropagate through the CNN layers, accumulating gradients for each layer's parameters.

## 3. Utility Functions

### 3.1. Activations (`utils/activations.py`)

- `relu`, `sigmoid`, `tanh`, `softmax`: Standard activation functions.
- `relu_prime`, `sigmoid_prime`, `tanh_prime`: Derivatives of the respective activation functions, used in backpropagation.

### 3.2. Data Loader (`utils/data_loader.py`)

- `download_mnist()`: Downloads the MNIST dataset if not already present.
- `load_mnist()`: Loads the MNIST images and labels from the gzipped files.
- `MNISTLoader`: A class that provides an iterable for batching and shuffling the dataset during training. It also normalizes image pixel values to the range [0, 1].

### 3.3. Losses (`utils/losses.py`)

- `cross_entropy`: Calculates the cross-entropy loss between predicted logits and one-hot encoded true labels.
- `cross_entropy_prime`: Calculates the derivative of the cross-entropy loss with respect to the logits.

### 3.4. Optimizers (`utils/optimizers.py`)

- `SGD`: Implements Stochastic Gradient Descent with an optional momentum term. The `step` method updates the model's parameters based on their gradients.

## 4. Training and Evaluation

### 4.1. Training (`train.py`)

- `one_hot`: A helper function to convert integer labels into one-hot encoded vectors.
- `train`: The main training loop.
    - Iterates over epochs.
    - For each epoch, it iterates through batches provided by the `MNISTLoader`.
    - Performs a forward pass to get logits.
    - Calculates the cross-entropy loss.
    - Performs a backward pass to compute gradients using `model.backward()`.
    - Updates model parameters using `optimizer.step()`.
    - Prints the average loss for each epoch.

### 4.2. Evaluation (`test.py`)

- `evaluate`: Evaluates the model's performance on a given dataset loader.
    - Iterates through batches of data.
    - Performs a forward pass to get logits.
    - Predicts the class by taking the argmax of the logits.
    - Compares predictions with true labels to count correct predictions.
    - Returns the overall accuracy.

## 5. Lessons Learned

1.  **Dimension Tracking is Crucial**: Meticulously track the dimensions of tensors at each step of the forward and backward passes. Even a single incorrect dimension can lead to cascading errors.
2.  **Understanding Layer Interactions**: Pay close attention to how layers interact, especially when combining different types of networks (e.g., CNN and LSTM). Ensure the output of one layer correctly matches the expected input of the next.
3.  **Debugging with Print Statements**: Strategic use of print statements to inspect tensor shapes and values at critical points in the code is invaluable for identifying where dimensions go awry.
4.  **Backward Pass Complexity**: The backward pass is often more complex than the forward pass, especially with custom layers. Thoroughly verify the gradient calculations and dimension consistency.
5.  **Caching for Backward Pass**: For recurrent networks like LSTMs, correctly caching intermediate values during the forward pass is essential for an accurate backward pass.
6.  **Iterative Debugging**: Debugging complex models is an iterative process. Isolate problems, test hypotheses, and progressively refine the code.

