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
from data.IAM import IAMPages, reconstruct_image, IAMPagesSplitted
from torch.amp import autocast, GradScaler

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


def loadBestModel(
    device="cuda",
) -> tuple[nn.Module, torch.optim.Optimizer, GradScaler, int]:
    from torch.cuda.amp import GradScaler

    bestModelPath, epochs = getBestModelPath()
    if bestModelPath is None:
        print(
            "[LineRemoverNN] [Model Loader] FATAL: No model found to load, exiting..."
        )
        exit(1)

    checkpoint = torch.load(bestModelPath, map_location=device)

    network = NeuralNetwork().to(device)
    network.load_state_dict(checkpoint["model_state"])

    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    # Ensure optimizer tensors are on the correct device
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    scaler = GradScaler("cuda")
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    print(
        f"[LineRemoverNN] [Model Loader] Loading {bestModelPath} which has {epochs + 1} epochs"
    )
    return network, optimizer, scaler, epochs


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")


def save_checkpoint(model, optimizer, scaler, epoch, path):
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
        },
        path,
    )
    print(f'[LineRemoverNN] [Trainer] Model saved at "{path}"')


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
    args = argsParser.parse_args()

    data_dir = args.dataset
    # Get all needed directories paths
    pages_dir = os.path.join(data_dir, "generated-pages")
    nolines_dir = os.path.join(data_dir, "generated-nolines-pages")
    json_dir = os.path.join(data_dir, "generated-words")
    pages_blocks_dir = os.path.join(data_dir, "generated-pages-blocks")
    nolines_blocks_dir = os.path.join(data_dir, "generated-nolines-pages-blocks")

    # Init network, dataset, dataloader, loss, and optimizer

    network = NeuralNetwork().to(device)
    dataset = IAMPagesSplitted(
        pages_blocks_dir,
        nolines_blocks_dir,
        "",
        readJson=False,
        transform=transforms.Compose(
            [
                # Instead of random rotation + random perspective : Random affine https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_illustrations.html#randomaffine
                # v2.RandomRotation(degrees=(0, 180)),
                #v2.RandomCrop(size=(128, 128)),
                # v2.RandomPerspective(distortion_scale=0.6, p=0.75),
                #v2.Resize(size=(512, 512)),
                v2.RandomResizedCrop(size=(512, 512), scale=(0.6, 1.0)),
                v2.ToDtype(torch.float32, scale=True),
            ]
        ),
        sameTransform=True,
    )
    dataloader = DataLoader(
        dataset, batch_size=10, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)
    epochs = args.epoch
    loss = combined_loss
    startingEpoch = 0
    network.apply(init_weights)

    # Create saved folder if not already created
    if not os.path.exists(os.path.join("./models", "saved")):
        os.mkdir(os.path.join("./models", "saved"))

    # Training...
    scaler = GradScaler(device)

    # Load model if specified
    if args.load:
        # Getting the bestModel
        network, optimizer, scaler, startingEpoch = loadBestModel()
        startingEpoch += 1  # We want to start the next epoch
        network.to(device)
    print(
        "[LineRemoverNN] [Trainer] Starting training... for",
        epochs - startingEpoch,
        "epochs",
    )

    torch.backends.cudnn.benchmark = True  # For speed improvement

    for epoch in range(startingEpoch, epochs):
        torch.cuda.empty_cache()
        network.train()
        totalLoss = 0

        for batch in tqdm(dataloader, desc=f"Epoch : {epoch}"):
            # Executing
            linesImages, noLines, shapes = batch
            linesImages, noLines = linesImages.to(device), noLines.to(device)

            optimizer.zero_grad()
            with autocast(device_type=device):
                processedFilters = network(linesImages)
            # Loss outside autocast to avoid issues with some loss functions
            # Compute loss between the subtracted image and the target (noLines)
            pixelLoss = loss(processedFilters.float(), noLines.float())

            # Back prog
            scaler.scale(pixelLoss).backward()
            scaler.step(optimizer)
            scaler.update()
            # pixelLoss.backward()
            # optimizer.step()

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
        save_checkpoint(
            network,
            optimizer,
            scaler,
            epoch,
            os.path.join(
                os.path.realpath(os.path.dirname(__file__)),
                f"models/saved/epoch-{epoch}.pt",
            ),
        )
        print(
            f'[LineremoverNN] [Trainer] Model saved at "./models/saved/epoch-{epoch}.pt"'
        )

    writer.flush()
