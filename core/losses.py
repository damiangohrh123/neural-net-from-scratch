import math
from typing import List

def cross_entropy_loss(predictions: List[List[float]], targets: List[List[float]]) -> float:
    """
    Compute Categorical Cross-Entropy Loss for a single sample.
    L = -sum(y_i * log(y_hat_i))

    Args:
        predictions: A 10x1 column vector (List[List]) of Softmax probabilities.
        targets: A 10x1 column vector (List[List]) of One-Hot encoded true labels.

    Returns:
        The scalar loss value for this specific sample.
    """
    loss = 0.0
    
    # Loop through the 10 classes (rows of the column vector)
    for i in range(len(predictions)):
        # targets[i][0] gets the float value from the [[val]] structure.
        # y_i is 1 only for the correct class (One-Hot).
        # We add a tiny epsilon (1e-15) to prevent log(0) which is undefined.
        if targets[i][0] > 0.5: # 0.5 is safer than 0 for floats
            loss -= math.log(predictions[i][0] + 1e-15)
                
    return loss

def cross_entropy_gradient(predictions: List[List[float]], targets: List[List[float]]) -> List[List[float]]:
    """
    Computes the gradient of the loss with respect to the output logits (y_hat - y).

    Args:
        predictions: A 10x1 column vector of Softmax probabilities.
        targets: A 10x1 column vector of One-Hot encoded true labels.

    Returns:
        A 10x1 column vector of gradients (errors) to be propagated backward.
    """
    gradient = []

    for i in range(len(predictions)):
        # Calculate the difference between prediction and truth (y_hat - y)
        # We wrap the result in [diff] to maintain the 10x1 matrix structure.
        diff = predictions[i][0] - targets[i][0]
        gradient.append([diff])

    return gradient