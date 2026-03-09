import random
from typing import List, Optional
from core.matrix_math import dot_product, add_matrices, transpose, hadamard_product, scalar_multiply, subtract_matrices
from core.activations import relu, relu_derivative, softmax

class Layer:
    """
    A single dense (fully connected) layer for a neural network.
    This class handles the linear transformation (Wx + b) of input data 
    and caches intermediate values to support backpropagation.

    Attributes:
        weights (List[List[float]]): Matrix of shape (output_size, input_size) representing connection strengths.
        biases (List[List[float]]): Matrix of shape (output_size, 1) representing the learned offset for each neuron.
        input_data (Optional[List[List[float]]]): A copy of the data that entered this layer.
        z (Optional[List[List[float]]]): The raw result of the math (before activation).
        output (Optional[List[List[float]]]): The final result of the layer (after activation).
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        """
        Initializes the Layer with random weights and zeroed biases.

        Args:
            input_size: The number of dimensions in the input vector.
            output_size: The number of neurons in this layer.
        """

        # self.weights is a list of lists (matrix) where each inner list represents the weights for a single neuron.:
        # [ [w1, w2, w3],  <- Neuron 1 (weights for inputs 1, 2, 3)
        #   [w1, w2, w3],  <- Neuron 2
        #   [w1, w2, w3] ] <- Neuron 3

        # Weights: matrix of (output_size x input_size)
        self.weights: List[List[float]] = []

        for row_index in range(output_size):
            neuron_weights: List[float] = []
            
            for col_index in range(input_size):
                # Generate a small random connection strength
                weight = random.uniform(-0.1, 0.1)
                neuron_weights.append(weight)
            
            self.weights.append(neuron_weights)

        # Biases: matrix of (output_size x 1)
        self.biases: List[List[float]] = [[0.0] for _ in range(output_size)]
        
        # Storage for backpropagation 
        self.input_data = None
        self.z = None # Pre-activation values
        self.output = None

    def forward(self, input_data: List[List[float]]) -> List[List[float]]:
        """
        Calculates the layer's result based on the input data.
        It multiplies the input by weights and adds the biases.
        The input and the result are saved for the training phase.

        Args:
            input_data: The list of numbers coming into this layer.

        Returns:
            The raw calculated result (z) (before activation).
        """
        self.input_data = input_data

        # Compute weighted sum
        weighted_sum = dot_product(self.weights, input_data)
        
        # Add bias
        self.z = add_matrices(weighted_sum, self.biases)
        
        return self.z

class NeuralNetwork:
    def __init__(self):
        """
        784 inputs -> 128 hidden (ReLU) -> 10 outputs (Softmax)
        """

        self.layers = [Layer(784, 128), Layer(128, 10)]

    def forward(self, x: List[List[float]]) -> List[List[float]]:
        """
        Performs a forward pass through the network architecture.
        Calculates the weighted sums and activations layer-by-layer:

        Args:
            x: The starting data (e.g., 784 image pixels).

        Returns:
            A list of 10 percentages showing how sure the network is 
            of each digit (0 to 9).
        """

        # Hidden Layer
        z1 = self.layers[0].forward(x)
        a1 = relu(z1)
        self.layers[0].output = a1

        # Output Layer
        z2 = self.layers[1].forward(a1)
        a2 = softmax(z2)
        self.layers[1].output = a2
        return a2

    def backward(self, loss_gradient: List[List[float]], learning_rate: float):
        """
        Executes the backpropagation algorithm to update network weights and biases.

        This method calculates how much each parameter contributed to the total error
        using the Chain Rule. It propagates the error signal from the output layer
        back to the hidden layer, then applies Gradient Descent to 'nudge' the 
        parameters in the direction that reduces loss.

        Args:
            loss_gradient: The error signal from the output layer (Predictions - Targets). Shape: (Batch Size x 10)
            learning_rate: A scalar that controls the step size of the update. Small values (0.01) ensure stable learning.
        """
        # Output Layer Error (delta2 = y_hat - y)
        delta2 = loss_gradient 
        
        # Hidden Layer Error (delta1)
        W2_T = transpose(self.layers[1].weights)
        error_signal1 = dot_product(W2_T, delta2)
        delta1 = hadamard_product(error_signal1, relu_derivative(self.layers[0].z))

        # Calculate Gradients for outptut layer
        dW2 = dot_product(delta2, transpose(self.layers[0].output))
        db2 = delta2

        # Calculate Gradients for hidden layer
        dW1 = dot_product(delta1, transpose(self.layers[0].input_data))
        db1 = delta1

        # Apply Gradient Descent: Update weights and biases by stepping in the opposite direction of the gradient
        self.layers[1].weights = subtract_matrices(self.layers[1].weights, scalar_multiply(dW2, learning_rate))
        self.layers[1].biases = subtract_matrices(self.layers[1].biases, scalar_multiply(db2, learning_rate))
        self.layers[0].weights = subtract_matrices(self.layers[0].weights, scalar_multiply(dW1, learning_rate))
        self.layers[0].biases = subtract_matrices(self.layers[0].biases, scalar_multiply(db1, learning_rate))