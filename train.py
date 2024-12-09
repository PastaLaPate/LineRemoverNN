import sys

sys.path.append("../")
import torch
from data.IAM import IAMPages, reconstruct_image, IAMPagesSplitted
from models.model import NeuralNetwork
from torch.utils.data import DataLoader
import torch.nn as nn
from mltu.utils.text_utils import ctc_decoder, get_cer
import torchvision.transforms as transforms

# from WordRecogniser.inf import infer
from argparse import ArgumentParser
from tqdm import tqdm
import os

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
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
    bestModelEpoch = 0
    for model in os.listdir("./models/saved/"):
        # epoch-x.pt
        modelEpoch = model[6:].split(".")[0]
        modelEpoch = int(modelEpoch)
        if modelEpoch > bestModelEpoch:
            bestModelEpoch = modelEpoch
            bestModelPath = f"./models/saved/epoch-{bestModelEpoch}.pt"
    return bestModelPath, bestModelEpoch

def loadBestModel():
    bestModelPath, epochs = getBestModelPath()
    network = NeuralNetwork()
    network.load_state_dict(torch.load(bestModelPath, weights_only=True))
    print(f"Loading {bestModelPath} which has {epochs} epochs")
    return network


if __name__ == "__main__":
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
        "-l", "--load", help="Load the best model", action="store_true", default=False
    )
    args = argsParser.parse_args()

    data_dir = args.dataset
    # Get all needed directories paths
    pages_dir = os.path.join(data_dir, 'generated-pages')
    nolines_dir = os.path.join(data_dir, 'generated-nolines-pages')
    json_dir = os.path.join(data_dir, 'generated-words')
    pages_blocks_dir = os.path.join(data_dir, 'generated-pages-blocks')
    nolines_blocks_dir = os.path.join(data_dir, 'generated-nolines-pages-blocks')
    
    # Init network, dataset, dataloader, loss, and optimizer

    network = NeuralNetwork().to(device)
    dataset = IAMPagesSplitted(pages_blocks_dir, nolines_blocks_dir, "", readJson=False)
    dataloader = DataLoader(
        dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=8
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
    epochs = args.epoch
    loss = nn.MSELoss()
    startingEpoch = 0
    
    # Load model if specified
    
    if args.load:
        # Getting the bestModel
        bestModelPath = None
        bestModelEpoch = 0
        for model in os.listdir('./models/saved/'):
            # epoch-x.pt
            modelEpoch = model[6:].split('.')[0]
            modelEpoch = int(modelEpoch)
            if modelEpoch > bestModelEpoch:
                bestModelEpoch = modelEpoch
                bestModelPath = f"./models/saved/epoch-{bestModelEpoch}.pt"
        
        if bestModelPath:
            print(f"Loading model path: {bestModelPath}")
            startingEpoch = bestModelEpoch + 1
            network.load_state_dict(torch.load(bestModelPath, weights_only=True))
    
    
    # Create saved folder if not already created
    if not os.path.exists(os.path.join('./models', 'saved')):
        os.mkdir(os.path.join('./models', 'saved'))
    
    # Training...
    

    for epoch in range(startingEpoch, epochs):
        torch.cuda.empty_cache()
        network.train()
        totalLoss = 0

        for batch in tqdm(dataloader):
            # Executing
            linesImages, noLines, shapes = batch
            linesImages, noLines = linesImages.to(device), noLines.to(device)
            processedFilters = network(linesImages)
            combinedFilter = torch.sum(processedFilters, dim=1, keepdim=True)  # Shape: [10, 1, 512, 512]

             # Ensure the combinedFilter is normalized if needed (optional)
            combinedFilter = combinedFilter / processedFilters.shape[1]

            # Subtract the composite filter from the input
            filteredImage = linesImages - combinedFilter  # Shape: [10, 1, 512, 512]
            
            # Compute loss between the subtracted image and the target (noLines)
            pixelLoss = loss(filteredImage, noLines)
            optimizer.zero_grad()
            # Back prog
            batch_loss = pixelLoss.clone().detach().requires_grad_(True)
            batch_loss.backward()
            optimizer.step()
            totalLoss += pixelLoss.item()
        print(f"Epoch : {epoch} Loss : ", totalLoss/len(dataloader))
        # Save each epoch
        torch.save(network.state_dict(), f"./models/saved/epoch-{epoch}.pt")
