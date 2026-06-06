from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision.transforms.v2 as v2
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader

from lineremovernn.commands.command import Command
from lineremovernn.data.pages import PagesDataset
from lineremovernn.model.model import LineRemovalUNet
from lineremovernn.utils import logging
from lineremovernn.utils.consts import DEFAULT_MODELS, DEVICE
from lineremovernn.utils.saver import (
    get_latest_model,
    load_model,
)

logger = logging.get_logger("ModelTester")


class TestCommand(Command):
    def __init__(self):
        super().__init__(
            name="test",
            description="Test the model.",
        )

    def init_parser(self, parser):
        parser.add_argument(
            "-n",
            "--n",
            type=int,
            default=5,
            help="Number of images to test for.",
        )
        parser.add_argument(
            "-b",
            "--batch-size",
            type=int,
            default=6,
            help="Batch size.",
        )
        parser.add_argument(
            "-d",
            "--dataset",
            type=Path,
            default=PagesDataset.path(),
            help="Training dataset's path.",
        )
        parser.add_argument(
            "-m",
            "--m",
            type=Path,
            default=DEFAULT_MODELS,
            help="Models location.",
        )

    def execute(self, args: Namespace) -> None:
        model = LineRemovalUNet().to(DEVICE)

        lm = get_latest_model()
        if lm is not None:
            latest_model = load_model(lm[1], training=False)
            model.load_state_dict(latest_model.model_state)
            logger.info(f"Loaded model weights from epoch {latest_model.stats.epoch}")
        else:
            logger.warning(
                "No saved model found. Running inference with random weights!"
            )

        model.eval()

        transforms = v2.Compose(
            (
                [
                    # STEP 1: Fast crop to a slightly larger intermediate size.
                    # Slashes canvas from 1.5M pixels down to ~147k pixels.
                    v2.RandomCrop(
                        (384, 384),
                        pad_if_needed=True,
                        fill=0,
                        padding_mode="constant",
                    ),
                    # STEP 2: Run heavy geometric warps on the tiny canvas.
                    # (Also fixed fill=255 to fill=0 to keep your black padding consistent!)
                    v2.RandomPerspective(distortion_scale=0.15, p=0.5, fill=0),
                    v2.RandomAffine(
                        degrees=(-1.5, 1.5),
                        scale=(
                            0.75,
                            1.3,
                        ),  # 384 * 0.75 = 288 (still safely larger than 256)
                        shear=(-4, 4),
                        fill=0,
                    ),
                    # STEP 3: Final cut to your model's exact input size.
                    v2.RandomCrop(
                        (256, 256),
                        pad_if_needed=True,
                        fill=0,
                        padding_mode="constant",
                    ),
                    v2.ToDtype(torch.float32, scale=True),
                ]
            )
        )

        dataset = PagesDataset(transforms)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        inputs = []
        preds = []
        ground_truths = []

        logger.info(f"Gathering {args.n} samples for visual inference...")
        with torch.no_grad():
            i = 0
            for r, c in dataloader:
                i += 1
                ruled = r.to(DEVICE)

                with autocast(DEVICE):
                    pred = model(ruled)
                    logger.info(f"Sample {i} generated.")

                ruled = ruled.cpu()
                pred = torch.clamp(pred.cpu(), 0, 1)
                clean = c.cpu()

                for i in range(ruled.size(0)):
                    if len(inputs) >= args.n:
                        break
                    inputs.append(ruled[i])
                    preds.append(pred[i])
                    ground_truths.append(clean[i])

                if len(inputs) >= args.n:
                    break

        fig, axes = plt.subplots(
            nrows=3, ncols=args.n, figsize=(3 * args.n, 9), squeeze=False
        )

        for col_idx in range(args.n):

            def tensor_to_np(tensor):
                np_img = tensor.permute(1, 2, 0).numpy()
                if np_img.shape[-1] == 1:  # If grayscale, squeeze the channel dimension
                    np_img = np_img.squeeze(-1)
                return np_img

            in_img = tensor_to_np(inputs[col_idx])
            pred_img = tensor_to_np(preds[col_idx])
            gt_img = tensor_to_np(ground_truths[col_idx])

            cmap = "gray" if in_img.ndim == 2 else None

            # Row 0: Input (Ruled line image)
            axes[0, col_idx].imshow(in_img, cmap=cmap)
            axes[0, col_idx].axis("off")
            if col_idx == 0:
                axes[0, col_idx].set_ylabel(
                    "Input (Ruled)", fontsize=14, fontweight="bold"
                )

            # Row 1: Model Prediction
            axes[1, col_idx].imshow(pred_img, cmap=cmap)
            axes[1, col_idx].axis("off")
            if col_idx == 0:
                axes[1, col_idx].set_ylabel(
                    "Prediction", fontsize=14, fontweight="bold"
                )

            # Row 2: Ground Truth (Clean image)
            axes[2, col_idx].imshow(gt_img, cmap=cmap)
            axes[2, col_idx].axis("off")
            if col_idx == 0:
                axes[2, col_idx].set_ylabel(
                    "Ground Truth", fontsize=14, fontweight="bold"
                )

        plt.tight_layout()
        logger.info("Displaying inference plot...")
        plt.show()
