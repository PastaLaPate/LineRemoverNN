
import sys
sys.path.append('../')
import torch
from data.IAM import IAMPages, reconstruct_image, IAMPagesSplitted
from models.model import NeuralNetwork
from torch.utils.data import DataLoader
import torch.nn as nn
from mltu.utils.text_utils import ctc_decoder, get_cer
import torchvision.transforms as transforms
#from WordRecogniser.inf import infer
from tqdm import tqdm
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def collate_fn(batch):
    # Extract and stack tensors for lines, noLines, etc.
    linesImages = torch.stack([item['lines'] for item in batch])
    noLines = torch.stack([item['noLines'] for item in batch])
    shapes = [item['shape'] for item in batch]
    return linesImages, noLines, shapes

pages_dir = '/mnt/c/users/alexa/DatasetData/generated-pages/'
nolines_dir = '/mnt/c/users/alexa/DatasetData/generated-nolines-pages/'
pages_blocks_dir = '/mnt/c/users/alexa/DatasetData/generated-pages-blocks/'
nolines_blocks_dir = '/mnt/c/users/alexa/DatasetData/generated-nolines-pages-blocks/'

if __name__ == '__main__':
    network = NeuralNetwork().to(device)
    dataset = IAMPagesSplitted(pages_blocks_dir, nolines_blocks_dir, '', readJson=False)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=collate_fn, num_workers=8)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
    epochs = 50
    loss = nn.MSELoss()
    for epoch in range(epochs):
        torch.cuda.empty_cache()
        network.train()
        totalLoss = 0
        
        for batch in tqdm(dataloader):
            linesImages, noLines, shapes = batch
            linesImages, noLines = linesImages.to(device), noLines.to(device)
            processedImages = network(linesImages)
        
            pixelLoss = loss(processedImages, noLines)
            optimizer.zero_grad()
            batch_loss = pixelLoss.clone().detach().requires_grad_(True)
            batch_loss.backward()
            optimizer.step()
            totalLoss += pixelLoss.item()
        print(f"Epoch : {epoch} Loss : ", totalLoss/len(dataloader))
        torch.save(network.state_dict(), f"./models/saved/epoch-{epoch}.pt")