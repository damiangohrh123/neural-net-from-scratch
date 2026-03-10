import math
from typing import List

def relu(Z: List[List[float]]) -> List[List[float]]:
    """
    If the value is negative, it becomes 0. If positive, it stays the same.
    """
    return [[max(0.0, val) for val in row] for row in Z]

def relu_derivative(Z: List[List[float]]) -> List[List[float]]:
    """
    Used during training to see which neurons were "on" (1.0) 
    and which were "off" (0.0).
    """
    return [[1.0 if val > 0 else 0.0 for val in row] for row in Z]

def softmax(Z: List[List[float]]) -> List[List[float]]:
    """
    Turns raw scores into percentages that sum to 100%.
    
    It makes the highest score stand out and ensures all 
    results are between 0 and 1, so they look like probabilities.
    """
    # Flatten the 10x1 to a simple list of 10 numbers to find the global max
    flat_z = [row[0] for row in Z]
    max_val = max(flat_z)

    # Calculate e^(z - max) for all 10 numbers
    exps = [math.exp(val - max_val) for val in flat_z]
    
    # Sum all 10 exponentials
    sum_exps = sum(exps)

    # Return as a 10x1 column vector again
    return [[e / sum_exps] for e in exps]