import os
from data.loader import get_data
from network.model import NeuralNetwork
from training.trainer import Trainer

def evaluate(model, dataset):
    """Helper function to calculate accuracy percentage."""
    correct = 0
    for x, y in dataset:
        prediction = model.forward(x)
        # prediction is [[p0], [p1]...[p9]]
        predicted_digit = max(range(10), key=lambda i: prediction[i][0])
        true_digit = max(range(10), key=lambda i: y[i][0])
        
        if predicted_digit == true_digit:
            correct += 1
    return (correct / len(dataset)) * 100

def main():
    base_path = "data/raw/"
    train_img = os.path.join(base_path, "train-images-idx3-ubyte.gz")
    train_lbl = os.path.join(base_path, "train-labels-idx1-ubyte.gz")
    test_img = os.path.join(base_path, "t10k-images-idx3-ubyte.gz")
    test_lbl = os.path.join(base_path, "t10k-labels-idx1-ubyte.gz")

    print("--- Loading MNIST Dataset ---")
    try:
        train_dataset = get_data(train_img, train_lbl)
        test_dataset = get_data(test_img, test_lbl)
    except FileNotFoundError:
        print("Error: MNIST files not found in data/raw/. Please download them first!")
        return

    print("--- Initializing Neural Network ---")
    mnist_model = NeuralNetwork()

    learning_rate = 0.01
    epochs = 5
    batch_size = 32
    best_accuracy = 0.0
    
    trainer = Trainer(mnist_model, learning_rate=learning_rate)

    print(f"--- Starting Training (Total Epochs: {epochs}) ---")

    # Progress checking
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        
        # Run the trainer for exactly 1 epoch
        trainer.train(train_dataset, epochs=1, batch_size=batch_size)

        # After each epoch, see how we are doing on the TEST data
        print("Evaluating...")
        accuracy = evaluate(mnist_model, test_dataset)
        print(f"Test Accuracy: {accuracy:.2f}%")

        # Save the best version
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            mnist_model.save_model("mnist_model_best.json")
            print(f"New best model saved with {accuracy:.2f}% accuracy!")

    # Final save for the "last" version
    mnist_model.save_model("mnist_model_final.json")
    print(f"\nTraining Complete! Best Accuracy achieved: {best_accuracy:.2f}%")

if __name__ == "__main__":
    main()