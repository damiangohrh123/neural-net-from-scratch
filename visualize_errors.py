import matplotlib.pyplot as plt
import os
from data.loader import get_data
from network.model import NeuralNetwork

def save_error_visual(test_data, index, true_label, pred_label, filename):
    # Flatten the 784x1 image back to 28x28 for plotting
    image_data = test_data[index][0]
    pixels = [row[0] for row in image_data]
    image_2d = [pixels[i:i+28] for i in range(0, 784, 28)]

    plt.figure(figsize=(4, 4))
    plt.imshow(image_2d, cmap='gray')
    plt.title(f"True: {true_label} | Predicted: {pred_label}", fontsize=12)
    plt.axis('off')
    
    if not os.path.exists('./images/errors'):
        os.makedirs('./images/errors')
        
    plt.savefig(f'./images/errors/{filename}.png', bbox_inches='tight')
    plt.close()
    print(f"Saved error case to ./images/errors/{filename}.png")

# --- EXECUTION ---
test_data = get_data("data/raw/t10k-images-idx3-ubyte.gz", "data/raw/t10k-labels-idx1-ubyte.gz")

nn = NeuralNetwork()
nn.load_model('mnist_model_best.json')

y_true = []
y_pred = []

print("Searching for specific error cases...")
for x, y_one_hot in test_data:
    # Extract digit from [[val]] format
    y_true.append([row[0] for row in y_one_hot].index(1.0))
    
    # Forward pass
    out = nn.forward(x)
    
    # Extract prediction from [[val]] format
    flat_out = [row[0] for row in out]
    y_pred.append(flat_out.index(max(flat_out)))

# Find a "9 predicted as 4"
idx_9as4 = next(i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t == 9 and p == 4)
save_error_visual(test_data, idx_9as4, 9, 4, "error_9_as_4")

# Find a "5 predicted as 3"
idx_5as3 = next(i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t == 5 and p == 3)
save_error_visual(test_data, idx_5as3, 5, 3, "error_5_as_3")

# Find a correct "1" 
idx_correct1 = next(i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t == 1 and p == 1)
save_error_visual(test_data, idx_correct1, 1, 1, "success_1")