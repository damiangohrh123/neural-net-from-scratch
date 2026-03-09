import math
from typing import List

def cross_entropy_loss(predictions: List[List[float]], targets: List[List[float]]) -> float:
    """
    Compute Categorical Cross-Entropy Loss.
    L = -sum(y_i * log(y_hat_i))

    Args:
        predictions: A 2D list (Batch Size x 10) of Softmax probabilities.
        targets: A 2D list (Batch Size x 10) of One-Hot encoded true labels.

    Returns:
        The average scalar loss across the entire batch.
    """
    total_loss = 0.0
    batch_size = len(predictions)
    
    for i in range(batch_size):
        for j in range(len(predictions[i])):
            # y_i is 1 only for the correct class (One-Hot)
            # We add a tiny epsilon (1e-15) to prevent log(0) which is undefined
            if targets[i][j] > 0:
                total_loss -= targets[i][j] * math.log(predictions[i][j] + 1e-15)
                
    return total_loss / batch_size

def cross_entropy_gradient(predictions: List[List[float]], targets: List[List[float]]) -> List[List[float]]:
    """
    Computes the gradient of the loss with respect to the output logits (y_hat - y).

    Args:
        predictions: A 2D list (Batch Size x 10) of Softmax probabilities.
        targets: A 2D list (Batch Size x 10) of One-Hot encoded true labels.

    Returns:
        A 2D list of gradients (errors) to be passed back through the network.
    """
    gradient = []
    
    for i in range(len(predictions)):
        row_grad = []
        for j in range(len(predictions[i])):
            # Calculate the difference between prediction and truth
            row_grad.append(predictions[i][j] - targets[i][j])
        gradient.append(row_grad)
    return gradient