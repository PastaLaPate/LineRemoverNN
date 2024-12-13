from train import loadBestModel
from torch import Tensor
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
from torchvision.transforms import ToTensor
from typing import List, Union
from data.IAM import split_into_blocks
import torch.cuda as torchc
from postProcessing import thresholdImage
import numpy as np
import cv2

device = "cuda" if torchc.is_available() else "cpu"

print("[LineRemoverNN] Loading model...")
network = loadBestModel()
network.eval()
network.to(device)


def processImg(img: Union[str, np.ndarray, Tensor], postProcess=True) -> np.ndarray:
    """
    ## Process an image and remove ruled lines
    ### Args:
      img: (``str | numpy.ndarray | torch.Tensor``)
        **img must be 512*512**
        - If image is ``str``, it loads it into a Tensor
        - If image is ``ndarray`` and not grayscale, converts it to grayscale and then to ``Tensor``
        - If image is ``ndarray`` and grayscale, converts it to ``Tensor``
        - If image is ``Tensor``, the image is directly moved to the best device.

      postProcess: (``bool``)
        - If ``True``, applies a thresholding to the image to have better contrast between text and background
    ### Returns:
      ``numpy.ndarray`` Grayscale image ndarray of shape [1, 512, 512] range 0<->255
    ### Raises:
      Exception: The image isn't 512x512
    ### Notes:
      - If the input image is not 512x512, you can use `LineRemoverNN.data.IAM.split_into_blocks` to preprocess it.
    ### Examples:
    >>> # Using str as path :
    >>> img = processImg('./Image-Path.png')
    >>> # Using RGB ndarray :
    >>> img = cv2.imread('./Image-Path.png')
    >>> img = processImg(img)
    >>> # Using GrayScale ndarray:
    >>> img = cv2.imread('./Image-Path.png', cv2.IMREAD_GRAYSCALE)
    >>> img = processImg(img)
    >>> # Using Tensor:
    >>> img = torchvision.io.read_image('./Image-Path.png', torchvision.io.image.ImageReadMode.GRAY)
    >>> img = processImg(img)

    """
    tensorimg: Tensor = Tensor()
    if isinstance(img, str):
        tensorimg = read_image(img, ImageReadMode.GRAY)
    if isinstance(img, np.ndarray):
        if not (len(img.shape) == 2 or img.shape[2] == 1):  # Is Image grayscale
            tensorimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tensorimg = ToTensor()(img)
    tensorimg = tensorimg.to(device)
    if not (
        tensorimg.shape[1] == 512 and tensorimg.shape[2] == 512
    ):  # Check if image is 512x512 (required size)
        raise Exception(
            "Provided image isn't 512x512 you may use LineRemoverNN.data.IAM.split_into_blocks function to split it"
        )
    imgs = tensorimg.unsqueeze(0)  # Add batch dimension
    outputs: Tensor = network(imgs)  # Process img
    output: Tensor = outputs.squeeze(0)  # Remove batch dimension
    output = output - tensorimg  # Apply filter
    output = output.detach().cpu()  # Detach and to cpu for conversion
    output = output * 255  # * 255 for normalisation between 0 and 255
    outputImage: np.ndarray = output.numpy()  # Convert to numpy aray
    if postProcess:
        outputImage = thresholdImage(outputImage)  # If postprocess, threshold the image

    return outputImage


def splitAndProcessImg(
    img: Union[str, np.ndarray], postProcess=True
) -> List[np.ndarray]:
    """
    ## Split and process an image using the `processImg` function
    Splits the input image into 512x512 blocks and processes each block to remove ruled lines.

    ### Args:
      img: (``str | numpy.ndarray``) The input image.
          - If ``str``, the path to the image. The image is loaded in grayscale.
          - If ``numpy.ndarray``, the image array in grayscale.

      postProcess: (``bool``)
          - See ``processImg.postProcess``

    ### Returns:
      ``List[numpy.ndarray]``: A list of processed grayscale image arrays, one for each 512x512 block.

    ### Examples:
    >>> # Using an image path:
    >>> blocks = splitAndProcessImg('./largeImage.png')
    >>> # Using a grayscale numpy image:
    >>> img = cv2.imread('./largeImage.png', cv2.IMREAD_GRAYSCALE)
    >>> blocks = splitAndProcessImg(img)
    """
    return processImgs(
        split_into_blocks(
            img
            if isinstance(img, np.ndarray)
            else cv2.imread(img, cv2.IMREAD_GRAYSCALE),
            block_size=512,
        ),
        postProcess=postProcess,
    )


def processImgs(
    imgs: List[Union[str, np.ndarray, Tensor]], postProcess=True
) -> List[np.ndarray]:
    """
    ## Process a list of images and remove ruled lines
    Applies `processImg` to each image in the provided list.

    ### Args:
      imgs: (``List[str | numpy.ndarray | torch.Tensor]``) The list of images to process.
      postProcess: (``bool``) Whether to apply thresholding to each processed image.

    ### Returns:
      ``List[numpy.ndarray]``: A list of processed grayscale image arrays.

    ### Examples:
    >>> # Using image paths:
    >>> imgs = ['./img1.png', './img2.png']
    >>> processed_imgs = processImgs(imgs)
    >>> # Using numpy images:
    >>> imgs = [cv2.imread('./img1.png'), cv2.imread('./img2.png')]
    >>> processed_imgs = processImgs(imgs)
    """
    return [processImg(img, postProcess=postProcess) for img in imgs]
