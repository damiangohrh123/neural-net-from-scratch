import random
from typing import List, Optional
from core.matrix_math import dot_product, add_matrices, transpose

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