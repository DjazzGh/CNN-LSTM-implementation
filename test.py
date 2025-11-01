import numpy as np

def evaluate(model, data_loader):
    """Evaluates the given model's accuracy on the test dataset.

    Args:
        model: The neural network model to be evaluated.
        data_loader: An instance of MNISTLoader to provide test data batches.

    Returns:
        float: The accuracy of the model on the test dataset (between 0 and 1).
    """
    correct_predictions = 0
    total_samples = 0
    num_batches = data_loader.get_num_test_batches()

    for i in range(num_batches):
        images, labels = data_loader.get_test_batch()

        # Normalize images to [0, 1] and reshape for CNN-LSTM input
        images = images.astype(np.float32) / 255.0
        # Reshape for CNN input (batch, height, width, channels)
        images = images.reshape(-1, 28, 28, 1)

        # Forward pass to get logits
        logits = model.forward(images)

        # Get predicted class (index of the highest logit)
        predictions = np.argmax(logits, axis=1)

        # Compare predictions with true labels
        correct_predictions += np.sum(predictions == labels)
        total_samples += len(labels)

    accuracy = correct_predictions / total_samples
    return accuracy