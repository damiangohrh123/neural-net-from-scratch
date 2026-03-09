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
    output = []
    for row in Z:
        # Compute exponentials (shifted by max_val to prevent math.exp overflow)
        # Every value is now <= 0, so math.exp stays between 0 and 1
        max_val = max(row)
        exps = [math.exp(val - max_val) for val in row]

        # Normalize so they sum to 1
        sum_exps = sum(exps)
        output.append([e / sum_exps for e in exps])
    return output