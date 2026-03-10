import os
import random
import matplotlib.pyplot as plt
from data.loader import get_data
from network.model import NeuralNetwork

def visualize_prediction(image_vector, true_label, prediction_vector):
    """
    Creates a side-by-side plot: The image and the probability bar chart.
    """
    # Prepare the image (reshape 784x1 to 28x28)
    img_data = [pixel[0] for pixel in image_vector]
    img_array = [img_data[i:i+28] for i in range(0, 784, 28)]
    
    # Prepare the probabilities
    probabilities = [p[0] for p in prediction_vector]
    predicted_digit = probabilities.index(max(probabilities))
    
    # Setup the Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Left side: The Actual Image
    ax1.imshow(img_array, cmap='gray')
    ax1.set_title(f"True Digit: {true_label}")
    ax1.axis('off')
    
    # Right side: The Confidence Chart
    colors = ['gray'] * 10
    colors[predicted_digit] = 'green' if predicted_digit == true_label else 'red'
    
    ax2.bar(range(10), probabilities, color=colors)
    ax2.set_xticks(range(10))
    ax2.set_ylim([0, 1.1])
    ax2.set_title("Model Confidence (%)")
    ax2.set_xlabel("Digits")
    
    plt.tight_layout()
    plt.show()

def main():
    # Load Model and Data
    model = NeuralNetwork()
    model.load_model("mnist_model_best.json")
    
    test_dataset = get_data("data/raw/t10k-images-idx3-ubyte.gz", 
                            "data/raw/t10k-labels-idx1-ubyte.gz")

    # Pick 1 random sample
    x, y = random.choice(test_dataset)
    
    # Predict
    prediction = model.forward(x)
    true_digit = y.index([1.0]) # Find which index is 1.0 in the one-hot list
    
    # Show the window
    visualize_prediction(x, true_digit, prediction)

if __name__ == "__main__":
    main()