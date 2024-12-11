import torch
from train import loadBestModel
from torchvision.io import read_image
import torchvision
import matplotlib.pyplot as plt
import cv2

if __name__ == "__main__":
    network = loadBestModel()
    network.eval()
    network.to("cuda")
    pathImageTest = "./data/generated-pages-blocks/0.png"

    # Load and preprocess the image
    img = (
        read_image(pathImageTest, torchvision.io.ImageReadMode.GRAY).float() / 255.0
    ).to("cuda")

    # Add batch dimension
    img = img.unsqueeze(0)  # Shape: (1, 1, H, W)

    # Pass the image through the network
    output = network(img)
    output_image = img - (
        torch.mean(output, dim=1, keepdim=True).squeeze()
    )  # Remove batch dimension, move to CPU, and convert to numpy
    output_image = output_image.squeeze().detach().cpu().numpy()

    # Display the image
    plt.imshow(output_image, cmap="gray")  # Use 'gray' for grayscale
    plt.axis("off")  # Hide axes for better visualization
    plt.show()
    cv2.imwrite("test.png", 255 * output_image)
