from train import loadBestModel
from torchvision.io import read_image
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

if __name__ == '__main__':
    network = loadBestModel()
    network.eval()
    network.to("cuda")
    pathImageTest = "./data/generated-pages-blocks/0.png"
    
    # Load and preprocess the image
    img = (read_image(pathImageTest, torchvision.io.ImageReadMode.GRAY).float() / 255.0).to("cuda")
    
    # Add batch dimension
    img = img.unsqueeze(0)  # Shape: (1, 1, H, W)
    
    # Pass the image through the network
    output = network(img)
    output_image = output.squeeze().detach().cpu().numpy()  # Remove batch dimension, move to CPU, and convert to numpy
    
    # Display the image
    plt.imshow(output_image, cmap="gray")  # Use 'gray' for grayscale
    plt.axis("off")  # Hide axes for better visualization
    plt.show()
