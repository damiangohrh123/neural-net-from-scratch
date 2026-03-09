import os
from data.loader import get_data
from network.model import NeuralNetwork
from training.trainer import Trainer

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
    # 784 -> 128 (ReLU) -> 10 (Softmax)
    mnist_model = NeuralNetwork()

    # Configure Trainer
    learning_rate = 0.01
    epochs = 5
    batch_size = 32
    
    trainer = Trainer(mnist_model, learning_rate=learning_rate)

    # Start Training
    print(f"--- Starting Training (Epochs: {epochs}, Batch Size: {batch_size}) ---")
    trainer.train(train_dataset, epochs=epochs, batch_size=batch_size)

    # Evaluation
    print("--- Evaluating Model ---")
    correct = 0
    for x, y in test_dataset:
        prediction = mnist_model.forward(x)
        
        # Get index of highest probability
        # prediction is a list of lists [[p0], [p1]...[p9]]
        predicted_digit = max(range(10), key=lambda i: prediction[i][0])
        true_digit = max(range(10), key=lambda i: y[i][0])
        
        if predicted_digit == true_digit:
            correct += 1

    accuracy = (correct / len(test_dataset)) * 100
    print(f"Final Test Accuracy: {accuracy:.2f}%")

    mnist_model.save_model("mnist_model_v1.json")

if __name__ == "__main__":
    main()