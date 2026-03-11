import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from data.loader import get_data 
from network.model import NeuralNetwork

def run_full_evaluation():
    # Setup paths
    test_img_path = "data/raw/t10k-images-idx3-ubyte.gz"
    test_lbl_path = "data/raw/t10k-labels-idx1-ubyte.gz"
    model_path = "mnist_model_best.json"

    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Train the model first!")
        return

    # Load the Dataset
    test_data = get_data(test_img_path, test_lbl_path) 

    # Initialize and Load the Model
    nn = NeuralNetwork()
    nn.load_model(model_path)

    y_true = []
    y_pred = []

    print(f"--- Running inference on {len(test_data)} test images ---")

    # Perform Inference
    for i, (image, label_one_hot) in enumerate(test_data):
        # Convert one-hot back to a single integer (0-9)
        # Using [row[0] for row in label_one_hot] because loader returns 10x1
        actual_digit = [row[0] for row in label_one_hot].index(1.0)
        y_true.append(actual_digit)
        
        # Forward pass
        output_probs = nn.forward(image)
        
        # Get index of highest probability
        predicted_digit = [row[0] for row in output_probs].index(max([row[0] for row in output_probs]))
        y_pred.append(predicted_digit)

        # Print progress every 1000 images
        if i % 1000 == 0:
            print(f"Processed {i} images...")

    # Generate and Save the Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    plt.title('Confusion Matrix Analysis', fontsize=14, pad=15)
    
    # Create directory if it doesn't exist
    if not os.path.exists('./images'):
        os.makedirs('./images')
        
    plt.tight_layout()
    plt.savefig('./images/confusion_matrix.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    run_full_evaluation()