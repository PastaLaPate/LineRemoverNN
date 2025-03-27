import sys

sys.path.append("../")

import math
import os
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from models.model import NeuralNetwork
from models.loss import *
from models.test_model import UNet
from data.IAM import IAMPages, reconstruct_image, IAMPagesSplitted

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def collate_fn(batch):
    # Extract and stack tensors for lines, noLines, etc.
    linesImages = torch.stack([item["lines"] for item in batch])
    noLines = torch.stack([item["noLines"] for item in batch])
    shapes = [item["shape"] for item in batch]
    return linesImages, noLines, shapes


data_dir = "./"


def getBestModelPath():
    bestModelPath = None
    bestModelEpoch = -1
    for model in os.listdir(
        os.path.join(os.path.realpath(os.path.dirname(__file__)), "models/saved/")
    ):
        # epoch-x.pt
        modelEpoch = model[6:].split(".")[0]
        modelEpoch = int(modelEpoch)
        if modelEpoch > bestModelEpoch:
            bestModelEpoch = modelEpoch
            bestModelPath = os.path.join(
                os.path.realpath(os.path.dirname(__file__)),
                f"models/saved/epoch-{bestModelEpoch}.pt",
            )
    return bestModelPath, bestModelEpoch


def loadBestModel() -> NeuralNetwork:
    bestModelPath, epochs = getBestModelPath()
    if bestModelPath is None:
        print(
            "[LineRemoverNN] [Model Loader] FATAL: No model found to load, exiting..."
        )
        exit(1)
    network = NeuralNetwork()
    network.load_state_dict(torch.load(bestModelPath, weights_only=True))
    print(
        f"[LineRemoverNN] [Model Loader] Loading {bestModelPath} which has {epochs+1} epochs"
    )
    return network


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")


if __name__ == "__main__":
    writer = SummaryWriter()
    argsParser = ArgumentParser(
        prog="LinerRemoverNN - Trainer", description="A trainer for the model"
    )
    argsParser.add_argument(
        "-e",
        "--epoch",
        type=int,
        help="The number of epoch to train the model on",
        required=True,
    )
    argsParser.add_argument("-d", "--dataset", help="Dataset location", required=True)
    argsParser.add_argument(
        "-l",
        "--load",
        help="Load the best model already saved before (continue training)",
        action="store_true",
        default=False,
    )
    argsParser.add_argument(
        "-lo",
        "--loss",
        required=True,
        help="Loss function to use : VGG, SSIM, L1 or BCE Loss",
        choices=["vgg", "ssim", "bce", "l1"],
    )
    args = argsParser.parse_args()

    data_dir = args.dataset
    # Get all needed directories paths
    pages_dir = os.path.join(data_dir, "generated-pages")
    nolines_dir = os.path.join(data_dir, "generated-nolines-pages")
    json_dir = os.path.join(data_dir, "generated-words")
    pages_blocks_dir = os.path.join(data_dir, "generated-pages-blocks")
    nolines_blocks_dir = os.path.join(data_dir, "generated-nolines-pages-blocks")

    # Init network, dataset, dataloader, loss, and optimizer

    network = UNet(1, 1).to(device)  # NeuralNetwork().to(device)
    dataset = IAMPagesSplitted(
        pages_blocks_dir,
        nolines_blocks_dir,
        "",
        readJson=False,
        transform=transforms.Compose(
            [
                # Instead of random rotation + random perspective : Random affine https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_illustrations.html#randomaffine
                # v2.RandomRotation(degrees=(0, 180)),
                v2.RandomCrop(size=(128, 128)),
                # v2.RandomPerspective(distortion_scale=0.6, p=0.75),
                v2.Resize(size=(512, 512)),
                v2.ToDtype(torch.float32, scale=True),
            ]
        ),
        sameTransform=True,
    )
    dataloader = DataLoader(
        dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=8
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    epochs = args.epoch
    loss = VGGLoss()
    l_name = "VGG"
    if args.loss == "vgg":
        loss = VGGLoss()
    elif args.loss == "ssim":
        loss = SSIM_L1_Loss()
        l_name = "SSIM"
    elif args.loss == "bce":
        loss = BCEL1Loss()
        l_name = "BCE"
    elif args.loss == "l1":
        loss = nn.L1Loss()
    print("[LineRemoverNN] [Trainer] Using loss function : ", l_name)
    startingEpoch = 0
    network.apply(init_weights)

    # Load model if specified

    if args.load:
        # Getting the bestModel
        network = loadBestModel().to(device)

    # Create saved folder if not already created
    if not os.path.exists(os.path.join("./models", "saved")):
        os.mkdir(os.path.join("./models", "saved"))

    # Training...
    print(
        "[LineRemoverNN] [Trainer] Starting training... for",
        epochs - startingEpoch,
        "epochs",
    )

    for epoch in range(startingEpoch, epochs):
        torch.cuda.empty_cache()
        network.train()
        totalLoss = 0

        for batch in tqdm(dataloader, desc=f"Epoch : {epoch}"):
            # Executing
            linesImages, noLines, shapes = batch
            linesImages, noLines = linesImages.to(device), noLines.to(device)

            optimizer.zero_grad()
            processedFilters = network(linesImages)

            # Compute loss between the subtracted image and the target (noLines)
            pixelLoss = loss(processedFilters, noLines)

            # Back prog
            pixelLoss.backward()
            optimizer.step()

            totalLoss += pixelLoss.item()

        avgLoss = totalLoss / len(dataloader)
        print(f"[LineRemoverNN] [Trainer] Epoch : {epoch}, Average Loss : {avgLoss}")
        print(f"[LineRemoverNN] [Trainer] Saving model...")
        writer.add_scalar("Loss/train", avgLoss, epoch)

        # --- Logging 5 sample output images to TensorBoard ---
        # Switch to evaluation mode to avoid randomness (like dropout or augmentation)
        network.eval()
        with torch.no_grad():
            # Get a sample batch from the dataloader (you might want to use a validation set)
            sample_batch = next(iter(dataloader))
            sample_linesImages, sample_noLines, sample_shapes = sample_batch
            sample_linesImages = sample_linesImages.to(device)
            sample_outputs = network(sample_linesImages)

            # Choose the first 5 outputs
            # If your images are in [N, C, H, W] format, ensure they're in the [0, 1] range.
            # Here we assume they are. Otherwise, you might need to apply a normalization.
            writer.add_images(
                "Processed Outputs",
                torch.clamp(sample_outputs[:5], 0, 1),
                epoch,
                dataformats="NCHW",
            )

        # Save each epoch
        torch.save(network.state_dict(), f"./models/saved/epoch-{epoch}.pt")
        print(
            f'[LineremoverNN] [Trainer] Model saved at "./models/saved/epoch-{epoch}.pt"'
        )

    writer.flush()
