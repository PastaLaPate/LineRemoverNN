import os
from torchvision.io import read_image
import torch
import json
from torch.utils.data import Dataset
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
import sys
import tqdm
import math
import numpy as np
from typing import List
from mltu.utils.text_utils import ctc_decoder, get_cer

sys.path.append("../../")
# from WordRecogniser.inf import infer

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def split_into_blocks(
    image: np.ndarray, block_size=512, transform=None
) -> List[np.ndarray]:
    """
    ## Split an ndarray into blocks of ``block_size``
    ### Params:
      image: The image to split ``ndarray`` Required
      block_size: The size of a block ``int`` (default: 512)
      transform: Eventual transform applied to each image ``func`` (default: None)
    ### Returns:
      List[numpy.ndarray] : the list of each block
    """
    # Get original image dimensions
    _, height, width = image.shape  # Assuming (C, H, W)

    # Calculate the padded dimensions
    padded_height = ((height + block_size - 1) // block_size) * block_size
    padded_width = ((width + block_size - 1) // block_size) * block_size

    # Pad the image with black (0 value)
    pad_bottom = padded_height - height
    pad_right = padded_width - width
    padded_image = F.pad(image, (0, pad_right, 0, pad_bottom), mode="constant", value=0)

    # Extract blocks
    blocks = []
    for i in range(0, padded_height, block_size):
        for j in range(0, padded_width, block_size):
            block = padded_image[:, i : i + block_size, j : j + block_size]
            if transform:
                block = transform(block)
            blocks.append(block)

    return blocks


def reconstruct_image(blocks, original_height, original_width, block_size=512):
    # Calculate padded dimensions
    padded_height = ((original_height + block_size - 1) // block_size) * block_size
    padded_width = ((original_width + block_size - 1) // block_size) * block_size

    # Create an empty tensor for the padded image
    _, channels, block_height, block_width = len(blocks), *blocks[0].shape
    reconstructed_padded_image = torch.zeros(
        (channels, padded_height, padded_width), dtype=blocks[0].dtype
    )

    # Iterate through blocks and reconstruct
    block_index = 0
    for i in range(0, padded_height, block_size):
        for j in range(0, padded_width, block_size):
            reconstructed_padded_image[:, i : i + block_size, j : j + block_size] = (
                blocks[block_index]
            )
            block_index += 1

    # Crop to the original dimensions
    reconstructed_image = reconstructed_padded_image[
        :, :original_height, :original_width
    ]
    return reconstructed_image


class IAMPages(Dataset):
    def __init__(
        self,
        lines_pages_dir,
        nolines_pages_dir,
        json_dir,
        random=False,
        splitSquare=False,
        transform=None,
        target_transform=None,
    ):
        self.lines_pages_dir = lines_pages_dir
        self.nolines_pages_dir = nolines_pages_dir
        self.json_dir = json_dir
        self.random = random
        self.splitSquare = splitSquare
        self.transform = transform
        self.target_transform = target_transform
        if not (
            len(os.listdir(self.lines_pages_dir))
            == len(os.listdir(self.nolines_pages_dir))
            == len(os.listdir(self.json_dir))
        ):
            raise Exception("Not same number of files in each directory")
        a = cv2.imread(os.path.join(self.lines_pages_dir, "0-page.png"))
        self.IMG_Height, self.IMG_WIDTH = a.shape[:2]
        self.blocksPerImage = math.ceil(self.IMG_Height / 512) * math.ceil(
            self.IMG_WIDTH / 512
        )

    def __len__(self):
        return len(os.listdir(self.lines_pages_dir) * self.blocksPerImage)

    def __getitem__(self, index):
        origIndex = index
        if self.splitSquare:
            index = index // self.blocksPerImage
        linesImagePath = os.path.join(self.lines_pages_dir, f"{index}-page.png")
        noLinesImagePath = os.path.join(self.nolines_pages_dir, f"{index}-page.png")
        jsonPath = os.path.join(self.json_dir, f"{index}.json")
        linesImage = read_image(linesImagePath)
        noLinesImage = read_image(noLinesImagePath)
        jsonRead = []
        with open(jsonPath, "r") as json_file:
            jsonRead = json.load(json_file)
        sample = {
            "lines": linesImage,
            "noLines": noLinesImage,
            "jsonData": jsonRead,
            "shape": linesImage.shape,
        }
        if self.splitSquare:
            linesImages = split_into_blocks(linesImage, transform=self.transform)[
                origIndex % self.blocksPerImage
            ]
            noLinesImages = split_into_blocks(noLinesImage, transform=self.transform)[
                origIndex % self.blocksPerImage
            ]
            sample = {
                "lines": linesImages,
                "noLines": noLinesImages,
                "jsonData": jsonRead,
                "shape": linesImage.shape,
            }
            return sample
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            sample = self.target_transform(sample)
        return sample


class IAMPagesSplitted(Dataset):
    def __init__(
        self,
        lines_pages_dir,
        nolines_pages_dir,
        json_dir,
        transform=None,
        target_transform=None,
        readJson=False,
    ):
        self.lines_pages_dir = lines_pages_dir
        self.nolines_pages_dir = nolines_pages_dir
        self.json_dir = json_dir
        self.transform = transform
        self.target_transform = target_transform
        self.readJson = readJson
        if not (
            os.path.exists(lines_pages_dir)
            or os.path.exists(nolines_pages_dir)
            or os.path.exists(json_dir)
        ):
            raise Exception("The provided folders do not exists")
        if not (
            len(os.listdir(self.lines_pages_dir))
            == len(os.listdir(self.nolines_pages_dir))
        ):
            raise Exception("Not same number of files in each directory")

    def __len__(self):
        return len(os.listdir(self.lines_pages_dir))

    def __getitem__(self, index):
        linesImagePath = os.path.join(self.lines_pages_dir, f"{index}.png")
        noLinesImagePath = os.path.join(self.nolines_pages_dir, f"{index}.png")
        jsonPath = os.path.join(self.json_dir, f"{index}.json")
        linesImage = (
            read_image(linesImagePath, torchvision.io.ImageReadMode.GRAY).float()
            / 255.0
        )
        noLinesImage = (
            read_image(noLinesImagePath, torchvision.io.ImageReadMode.GRAY).float()
            / 255.0
        )
        jsonRead = []
        if self.readJson:
            with open(jsonPath, "r") as json_file:
                jsonRead = json.load(json_file)
        if self.transform:
            linesImage = self.transform(linesImage)
            noLinesImage = self.transform(noLinesImage)
        sample = {
            "lines": linesImage,
            "noLines": noLinesImage,
            "jsonData": jsonRead,
            "shape": linesImage.shape,
        }
        if self.target_transform:
            sample = self.target_transform(sample)
        return sample


if __name__ == "__main__":
    pages_dir = "./generated-pages/"
    nolines_dir = "./generated-nolines-pages/"
    json_dir = "./generated-words/"
    dataset = IAMPages(
        pages_dir, nolines_dir, json_dir, random=False, splitSquare=False
    )
    cerLines = 0
    cerNoLines = 0
    wordsN = 0
    for i in tqdm.tqdm(range(50)):
        sample = dataset[i]
        img = sample["noLines"].permute(1, 2, 0).numpy()
        linesImg = sample["lines"].permute(1, 2, 0).numpy()
        jsonData = sample["jsonData"]
        wordsN += len(jsonData)
        for word in jsonData:
            x = word["x"]
            y = word["y"]
            w = word["w"]
            h = word["h"]

            slicedLines = linesImg[y : y + h, x : x + w]
            slicedNoLines = img[y : y + h, x : x + w]
            # predictedLines = infer(slicedLines)
            # predictedNoLines = infer(slicedNoLines)
            # cerLines += get_cer(predictedLines, word['text'])
            # cerNoLines += get_cer(predictedNoLines, word['text'])
            # cv2.putText(img, predictedNoLines, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5, cv2.LINE_AA)
            # cv2.putText(linesImg, predictedLines, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5, cv2.LINE_AA)
            # cv2.rectangle(img, (x, y), (x+w, y+h), color=(255, 0, 0), thickness=3)
            # cv2.rectangle(linesImg, (x, y), (x+w, y+h), color=(255, 0, 0), thickness=3)
        # plt.subplot(1, 2, 1)
        # plt.title(f"Line Image Avg CER : {(cerLines/len(jsonData))*100}%")
        # plt.imshow(linesImg)
        # plt.subplot(1, 2, 2)
        # plt.title(f"No lines image Avg CER : {(cerNoLines/len(jsonData))*100}%")
        # plt.imshow(img)
        # plt.show()
    print(
        f"Lines image Avg CER : {(cerLines/wordsN)*100}% No lines image Avg CER : {(cerNoLines/wordsN)*100}%"
    )
