import random
from typing import List, Optional
from core.matrix_math import dot_product, add_matrices, transpose

class Layer:
    def __init__(self, input_size: int, output_size: int) -> None:
        """
        Initialize weights and biases.
        input_size: Number of incoming lines (weights) per neuron.
        output_size: Number of neurons (nodes) in this layer.
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
        Takes a Matrix and returns a Matrix.
        """
        self.input_data = input_data
        
        # Compute weighted sum
        weighted_sum = dot_product(self.weights, input_data)
        
        # Add bias
        self.z = add_matrices(weighted_sum, self.biases)
        
        return self.z