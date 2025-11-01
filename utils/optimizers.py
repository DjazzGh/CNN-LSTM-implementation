import numpy as np


class SGD:
    """Stochastic Gradient Descent (SGD) optimizer with momentum.

    This optimizer updates model parameters using the gradient of the loss function
    with respect to the parameters, incorporating a momentum term to accelerate
    convergence and dampen oscillations.
    """
    def __init__(self, learning_rate=0.01, momentum=0.9):
        """Initializes the SGD optimizer.

        Args:
            learning_rate (float, optional): The step size for parameter updates. Defaults to 0.01.
            momentum (float, optional): The momentum factor. Defaults to 0.9.
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}

    def step(self, parameters, grads):
        """Updates the model parameters using SGD with momentum.

        Args:
            parameters (dict): A dictionary of parameter names and their corresponding numpy arrays.
            grads (dict): A dictionary of gradient names and their corresponding numpy arrays.
        """
        for param_name, param_value in parameters.items():
            grad_value = grads[param_name]

            if param_name not in self.velocities:
                self.velocities[param_name] = np.zeros_like(param_value)

            # Update velocity
            self.velocities[param_name] = self.momentum * self.velocities[param_name] + (1 - self.momentum) * grad_value

            # Update parameter
            parameters[param_name] -= self.learning_rate * self.velocities[param_name]