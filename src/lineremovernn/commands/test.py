from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader

from lineremovernn.commands.command import Command
from lineremovernn.data.pages import (
    EXTENDED_TRANSFORM,
    STANDARD_TRANSFORMS,
    PagesDataset,
    collate_pages,
)
from lineremovernn.model.lineremover import LineRemovalUNet
from lineremovernn.utils import logging
from lineremovernn.utils.consts import DEFAULT_MODELS, DEVICE, USE_AMP
from lineremovernn.utils.loss import criterion
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
            "-e",
            "--extended",
            action="store_true",
            help="Extended dataset transforms.",
        )
        parser.add_argument(
            "-l",
            "--loss",
            action="store_true",
            help="Show loss with target.",
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

        dataset = PagesDataset(
            STANDARD_TRANSFORMS if not args.extended else EXTENDED_TRANSFORM
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_pages,
        )

        inputs = []
        preds = []
        ground_truths = []
        losses: list[float] = []

        logger.info(f"Gathering {args.n} samples for visual inference...")
        with torch.no_grad():
            for r, c, p in dataloader:
                ruled = r.to(DEVICE)

                with autocast(DEVICE, dtype=torch.float16, enabled=USE_AMP):
                    pred = model(ruled)

                ruled = ruled.cpu()
                pred = torch.clamp(pred.cpu(), 0, 1)
                clean = c.cpu()

                for i in range(ruled.size(0)):
                    if len(inputs) >= args.n:
                        break
                    inputs.append(ruled[i])
                    preds.append(pred[i])
                    ground_truths.append(clean[i])
                    if args.loss:
                        loss = criterion(pred[i], clean[i])
                        losses.append(loss.item())
                        logger.info(f"Sample {i} generated. Loss : {loss.item()}")
                    else:
                        logger.info(f"Sample {i} generated.")

                if len(inputs) >= args.n:
                    break

        _fig, axes = plt.subplots(
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
            axes[0, col_idx].set_title(f"In {col_idx + 1}", fontsize=12)
            axes[0, col_idx].imshow(in_img, cmap=cmap)
            axes[0, col_idx].axis("off")
            if col_idx == 0:
                axes[0, col_idx].set_ylabel(
                    "Input (Ruled)", fontsize=14, fontweight="bold"
                )

            # Row 1: Model Prediction
            axes[1, col_idx].set_title(
                f"Pred {col_idx + 1} Loss : {f'{losses[col_idx]:.4f}'}"
                if args.loss
                else f"Pred {col_idx + 1}",
                fontsize=12,
            )
            axes[1, col_idx].imshow(pred_img, cmap=cmap)
            axes[1, col_idx].axis("off")
            if col_idx == 0:
                axes[1, col_idx].set_ylabel(
                    "Prediction",
                    fontsize=14,
                    fontweight="bold",
                )

            # Row 2: Ground Truth (Clean image)
            axes[2, col_idx].set_title(f"Target {col_idx + 1}", fontsize=12)
            axes[2, col_idx].imshow(gt_img, cmap=cmap)
            axes[2, col_idx].axis("off")
            if col_idx == 0:
                axes[2, col_idx].set_ylabel(
                    "Ground Truth", fontsize=14, fontweight="bold"
                )

        plt.tight_layout()
        logger.info("Displaying inference plot...")
        plt.show()
