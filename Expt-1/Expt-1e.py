"""Write a function that performs the sigmoid activation on a numpy array representing a layer's output in a neural network. The sigmoid function is defined as: sigmoid(x)= 1+e −x The function should be able to handle both scalar and numpy array inputs."""
import numpy as np
def sigmoid(x):
    """
    Compute the sigmoid function for a scalar or a numpy array.
    Parameters: x (scalar or numpy array): Input value or array of values.
    Returns: scalar or numpy array: Sigmoid of the input value or array.
    """
    return 1 / (1 + np.exp(-x))
# Example usage:
array_input = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
scalar_input = 0.5
# Applying the sigmoid function
sigmoid_array = sigmoid(array_input)
sigmoid_scalar = sigmoid(scalar_input)
print("Sigmoid of scalar input:", sigmoid_scalar)
print("Sigmoid of array input:")
print(sigmoid_array)

