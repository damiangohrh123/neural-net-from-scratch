import gradio as gr
import numpy as np
from PIL import ImageOps
import PIL.Image as PILImage
from network.model import NeuralNetwork

# Initialize and load model
nn = NeuralNetwork()
nn.load_model('mnist_model_best.json')

def predict_digit(image):
    if image is None:
        return None
    
    # Get the image (Gradio 5 uses a dict with 'composite')
    img_data = image['composite'] if isinstance(image, dict) else image
    
    # Convert to PIL, Grayscale, and Resize to 28x28
    pil_img = PILImage.fromarray(img_data).convert('L').resize((28, 28))
    
    # Check if the background is light (avg > 127). If so, we invert it.
    if np.mean(np.array(pil_img)) > 127:
        pil_img = ImageOps.invert(pil_img)
    
    # Final normalization and shaping
    img_array = np.array(pil_img).reshape(784, 1) / 255.0
    input_data = img_array.tolist()

    # Inference
    output = nn.forward(input_data)
    probabilities = [row[0] for row in output]
    
    return {str(i): float(probabilities[i]) for i in range(10)}

interface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(label="Draw a Digit (0-9)", type="numpy"),
    outputs=gr.Label(num_top_classes=3, label="Top Predictions"),
    title="Neural Network From Scratch",
    description="Draw a digit! The model will automatically invert and process the image.",
    live=True
)

if __name__ == "__main__":
    interface.launch()