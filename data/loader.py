import gzip
import struct
from typing import List, Tuple

def load_mnist_images(filename: str) -> List[List[List[float]]]:
    """
    Load MNIST images from binary file.
    Normalizes pixels to [0, 1] and flattens to 784x1 matrices.

    Args:
        filename: Path to the .gz file containing MNIST image data.

    Returns:
        A list of images, where each image is a 784x1 column vector (List[List[float]]).
    """

    with gzip.open(filename, 'rb') as f:
        # Read header (16 bytes): Magic Number, Number of Images, Rows, Columns
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        
        images = []
        for _ in range(count):
            # Read 784 bytes (one image)
            raw_pixels = f.read(rows * cols)

            # Normalize to [0.0, 1.0] and reshape into a 784x1 column vector for dot product operations.
            normalized_pixels = [[pixel / 255.0] for pixel in raw_pixels]
            images.append(normalized_pixels)
            
    return images

def load_mnist_labels(filename: str) -> List[List[List[float]]]:
    """
    Load MNIST labels and convert to One-Hot Encoding.

    Args:
        filename: Path to the .gz file containing MNIST label data.

    Returns:
        A list of 10x1 column vectors representing the 'True' probability distribution for each image.
    """

    with gzip.open(filename, 'rb') as f:
        # Read header (8 bytes): Magic Number and Number of Items
        magic, count = struct.unpack(">II", f.read(8))
        
        labels = []
        for _ in range(count):
            # Read 1 byte: The actual digit value (0-9)
            label_val = struct.unpack(">B", f.read(1))[0]

            # Create One-Hot Vector
            one_hot = [[0.0] for _ in range(10)]
            one_hot[label_val][0] = 1.0
            labels.append(one_hot)
            
    return labels

def get_data(img_path: str, lbl_path: str) -> List[Tuple[List[List[float]], List[List[float]]]]:
    """
    Helper to pair images and labels into a dataset list.

    Args:
        img_path: Path to the MNIST image .gz file.
        lbl_path: Path to the MNIST label .gz file.

    Returns:
        A list of tuples, where each tuple contains:
            - A 784x1 image matrix (Input X)
            - A 10x1 one-hot label matrix (Target Y)
    """

    images = load_mnist_images(img_path)
    labels = load_mnist_labels(lbl_path)

    return list(zip(images, labels))