import io
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def get_img_bits(image):
    """
    Converts PIL Image it to bits
    """
    bio = io.BytesIO()
    image.save(bio, format="PNG")
    return bio.getvalue()


def plot_image(image, title=None):
    """
    Displays an image using matplotlib
    """
    plt.figure(figsize=(8, 8))
    if title is not None:
        plt.title(title)
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def load_numpy(path, device):
    """
    Loads numpy array and converts it to torch tensor.
    """
    # Load from path
    data = np.load(path)
    # Convert to torch tensor and pass to device
    data = torch.tensor(data, device=device)
    return data


def convert_image(image, size):
    """
    Converts generated image to a displayable format.
    @param image: a generated image of shape [N, H, W] in range [-1, 1]
    @return: postprocessed image in shape [H, W, N] in range [0, 255] (dtype=uint8)
    """
    # Change the order of channels from [N, H, W] to [H, W, N]
    image = image.permute(1, 2, 0).cpu().numpy()

    # Renormalize image from [-1, 1] to [0, 255]
    image = (image + 1) * 127.5
    image = np.clip(image, 0, 255).astype(np.uint8)

    # Convert to PIL
    image = Image.fromarray(image)

    # Resize to a desired size
    image = image.resize(size)

    return image
