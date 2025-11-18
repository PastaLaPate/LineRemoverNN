import torch
from train import loadBestModel
from torchvision.io import read_image
import torchvision.transforms.v2 as v2
import torchvision
import matplotlib.pyplot as plt
from postProcessing import thresholdImage
from random import randint
from torchvision.transforms import ToTensor
from models.loss import *
import argparse
import cv2
import torch.nn as nn
import os
import time
import gc  # Garbage collector


if __name__ == "__main__":
    argParser = argparse.ArgumentParser(
        "Model Tester",
        description="Loads best model and tests a part of the dataset on it",
    )
    argParser.add_argument("-d", "--data", help="Path of the dataset", required=True)
    argParser.add_argument(
        "-p",
        "--postprocess",
        help="Should the images be post-processed after being treated",
        default=False,
        action="store_true",
    )
    argParser.add_argument(
        "-s",
        "--show",
        help="Show the treated images",
        action="store_true",
        default=False,
    )
    argParser.add_argument(
        "-n", "--n", help="Number of images to treat", default=10, type=int
    )
    argParser.add_argument(
        "-b", "--batch", help="Batch size for processing images", default=1, type=int
    )
    argParser.add_argument(
        "-l",
        "--loss",
        action="store_true",
        help="Print or not the loss of the treated images",
        default=False,
    )
    argParser.add_argument(
        "-r", "--random", action="store_true", help="Random page number", default=True
    )
    args = argParser.parse_args()

    torch.cuda.empty_cache()

    network = loadBestModel()[0]
    network.eval()
    network.to("cuda")

    numberOfImagesToTest = args.n
    batchSize = args.batch
    imgs = []
    noLinesImgs = []
    start = randint(
        1,
        (
            (
                len(os.listdir(os.path.join(args.data, "generated-pages-blocks")))
                - numberOfImagesToTest
            )
            if args.random
            else 2
        ),
    )
    if args.show:
        plt.figure(figsize=(numberOfImagesToTest * 3, 15))  # width scales with N

    transforms = v2.Compose([v2.RandomResizedCrop(size=(512, 512), scale=(0.6, 1.0)), v2.ToDtype(torch.float32, scale=True)])
    print("[LineRemoverNN] [Tester] Loading images...")
    for i in range(start, start + numberOfImagesToTest):
        pathImageTest = os.path.join(args.data, f"generated-pages-blocks/{i}.png")
        pathImageNoLineTest = os.path.join(
            args.data, f"generated-nolines-pages-blocks/{i}.png"
        )
        img = read_image(pathImageTest, torchvision.io.ImageReadMode.GRAY)
        imgNoLine = read_image(pathImageNoLineTest, torchvision.io.ImageReadMode.GRAY)
        if args.show:
            plt.subplot(5, numberOfImagesToTest, (i - start) + 1)
            plt.title(f"Lines {(i - start)}")
            plt.imshow(img.squeeze().numpy(), cmap="gray")
            plt.subplot(
                5, numberOfImagesToTest, numberOfImagesToTest * 4 + (i - start) + 1
            )
            plt.title(f"Goal {(i - start)}")
            plt.imshow(imgNoLine.squeeze().numpy(), cmap="gray")
        imgs.append(img.float() / 255.0)
        noLinesImgs.append(imgNoLine.float() / 255.0)

    imgs = torch.stack(imgs)
    noLinesImgs = torch.stack(noLinesImgs)

    print("[LineRemoverNN] [Tester] Using RMSE Loss...")
    loss = combined_loss

    startedTime = time.time_ns()
    print("[LineRemoverNN] [Tester] Treating images...")
    totalLoss = 0

    for batchStart in range(0, len(imgs), batchSize):
        batchEnd = min(batchStart + batchSize, len(imgs))
        batchImgs = imgs[batchStart:batchEnd].to("cuda")
        batchNoLinesImgs = noLinesImgs[batchStart:batchEnd].to("cuda")
        outputs = network(batchImgs)
        for idx, outputImg in enumerate(outputs):
            imgIdx = batchStart + idx
            print(batchImgs[idx].min(), batchImgs[idx].max())
            print(outputImg.min(), outputImg.max())
            print(batchNoLinesImgs[idx].min(), batchNoLinesImgs[idx].max())
            final = (outputImg.detach().cpu().squeeze() * 255).numpy()
            _in = (batchImgs[idx].detach().cpu().squeeze() * 255).numpy()

            if args.show:
                plt.subplot(5, numberOfImagesToTest, numberOfImagesToTest + imgIdx + 1)
                plt.title(f"Detected {imgIdx}")
                plt.imshow(final, cmap="gray")
                plt.subplot(
                    5, numberOfImagesToTest, numberOfImagesToTest * 2 + imgIdx + 1
                )
                plt.title(f"Output {imgIdx}")
                plt.imshow(final, cmap="gray")
                plt.subplot(
                    5, numberOfImagesToTest, numberOfImagesToTest * 3 + imgIdx + 1
                )
                plt.title(f"Final {imgIdx}")
                plt.imshow(
                    thresholdImage(final) if args.postprocess else final, cmap="gray"
                )
            pixelLoss = loss(outputImg.unsqueeze(0), batchNoLinesImgs[idx].unsqueeze(0))
            postProcessed = thresholdImage(final) if args.postprocess else final
            tensorPostProcessed = (
                ToTensor()(postProcessed).to("cuda").permute(0, 2, 1).unsqueeze(0)
            )

            pPLoss = loss(tensorPostProcessed, batchNoLinesImgs[idx].unsqueeze(0))
            pPlos = 10

            if args.loss:
                print(
                    f"[LineRemoverNN] [Tester] Image {imgIdx} loss : {pixelLoss.item()} final loss : {5}"
                )
                totalLoss += pixelLoss.item()
            del pixelLoss
            #del tensorPostProcessed, pixelLoss, pPLoss, postProcessed
            torch.cuda.empty_cache()

        del batchImgs, batchNoLinesImgs, outputs
        torch.cuda.empty_cache()
        gc.collect()

    elapsedTime = time.time_ns() - startedTime
    elapsedTimeSeconds = elapsedTime / 1e9
    print(
        f"[LineRemoverNN] [Tester] Treated {len(imgs)} images in {elapsedTimeSeconds:.2f} seconds"
    )

    if args.loss:
        print(f"[LineRemoverNN] [Tester] Avg loss : {totalLoss / len(imgs):.6f}")

    if args.show:
        plt.subplots_adjust(hspace=0.5)
        plt.show()
