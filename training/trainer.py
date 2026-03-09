import random
from typing import List, Tuple
from network.model import NeuralNetwork
from core.losses import cross_entropy_gradient, cross_entropy_loss

class Trainer:
    def __init__(self, network: NeuralNetwork, learning_rate: float = 0.01):
        """
        Initialize with a network and learning rate.

        Args:
            network: An instance of the NeuralNetwork class to be trained.
            learning_rate: The scalar (η) used to scale gradients during backpropagation. Defaults to 0.01.
        """
        self.network = network
        self.learning_rate = learning_rate

    def train(self, dataset: List[Tuple], epochs: int, batch_size: int):
        """
        The main training loop iterating over epochs and batches.

        Args:
            dataset: A list of (input, target) pairs from get_data().
            epochs: Total number of times to iterate over the entire dataset.
            batch_size: The number of samples processed before reporting average loss.
        """
        for epoch in range(epochs):
            # Shuffle to prevent the model from learning the sequence of the data.
            random.shuffle(dataset)
            epoch_loss = 0.0
            
            # Process data in Mini-Batches
            for i in range(0, len(dataset), batch_size):
                # Slice the dataset into a sub-list of size batch_size.
                batch = dataset[i : i + batch_size]

                # Perform forward/backward passes and accumulate the loss.
                batch_loss = self.train_mini_batch(batch)
                epoch_loss += batch_loss

                # Progress tracker
                if (i // batch_size) % 100 == 0:
                    print(f"Epoch {epoch+1} | Batch {i // batch_size} | Current Batch Loss: {batch_loss:.4f}")

            avg_epoch_loss = epoch_loss / (len(dataset) / batch_size)
            print(f"Epoch {epoch + 1}/{epochs} completed. Avg Loss: {avg_epoch_loss:.4f}")

    def train_mini_batch(self, batch: List[Tuple]) -> float:
        """
        Processes a small subset of the dataset to perform weight updates.

        For each sample in the batch, it performs a forward pass to generate predictions, calculates the 
        cross-entropy loss, and executes backpropagation to refine model parameters.

        Args:
            batch: A sub-list of (input, target) tuples.

        Returns:
            The average loss across all samples in this mini-batch.
        """

        total_loss = 0.0

        # Iterate through each image (x) and its one-hot label (y) in the batch
        for x, y in batch:
            # Forward Pass
            predictions = self.network.forward(x)

            # Compute Loss
            total_loss += cross_entropy_loss(predictions, y)

            # Initial Error Gradient
            loss_grad = cross_entropy_gradient(predictions, y)

            # Backpropagation & Update
            self.network.backward(loss_grad, self.learning_rate)

        return total_loss / len(batch)