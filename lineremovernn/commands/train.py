import time
from argparse import Namespace
from pathlib import Path

import torch
import torchvision.transforms.v2 as v2
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lineremovernn.commands.command import Command
from lineremovernn.data.pages import PagesDataset
from lineremovernn.model.model import LineRemovalUNet
from lineremovernn.utils import logging
from lineremovernn.utils.consts import DEFAULT_MODELS, DEVICE
from lineremovernn.utils.loss import criterion
from lineremovernn.utils.saver import (
    ModelStats,
    SavedModel,
    get_latest_model,
    load_model,
    save_model,
)

logger = logging.get_logger("ModelTrainer")


class TrainCommand(Command):
    def __init__(self):
        super().__init__(
            name="train",
            description="Train the model.",
        )

    def init_parser(self, parser):
        parser.add_argument(
            "-e",
            "--epoch",
            type=int,
            default=25,
            help="Number of epochs to train the model for.",
        )
        parser.add_argument(
            "-b",
            "--batch-size",
            type=int,
            default=6,
            help="Batch size.",
        )
        parser.add_argument(
            "-l",
            "--load",
            action="store_true",
            help="Load a previously trained model to continue training.",
        )
        parser.add_argument(
            "-ex",
            "--extended",
            action="store_true",
            help="Uses extended augmentations.",
        )
        parser.add_argument(
            "-d",
            "--dataset",
            type=Path,
            default=PagesDataset.path(),
            help="Training dataset's path.",
        )
        parser.add_argument(
            "-o",
            "--output",
            type=Path,
            default=DEFAULT_MODELS,
            help="Output location.",
        )

    def execute(self, args: Namespace) -> None:
        cudnn.benchmark = True

        args.epoch = args.epoch - 1

        model = LineRemovalUNet().to(DEVICE)
        current_epoch = -1
        latest_model: None | SavedModel = None
        if args.load:
            lm = get_latest_model()
            if lm is not None:
                latest_model = load_model(lm[1])
                current_epoch = latest_model.stats.epoch

        transforms = (
            v2.Compose(
                [
                    v2.RandomCrop((256, 256)),
                    v2.RandomRotation((-0.5, 0.5)),
                    v2.ToDtype(torch.float32, scale=True),
                ]
            )
            if not args.extended
            else v2.Compose(
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
        dataloader: DataLoader[PagesDataset] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        steps_per_epoch = len(dataloader)
        total_steps = steps_per_epoch * args.epoch

        optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim,
            T_max=total_steps,
            eta_min=1e-6,
        )
        if latest_model:
            model.load_state_dict(latest_model.model_state)
            optim.load_state_dict(latest_model.optim_state)
            scheduler.load_state_dict(latest_model.scheduler_state)

        logger.info(
            f"Starting training from epoch {current_epoch} for {args.epoch - current_epoch} epochs"
        )
        best_loss = float("inf")

        scaler = GradScaler(DEVICE)
        for e in range(current_epoch + 1, args.epoch + 1):
            start = time.time_ns()
            logger.info("Starting epoch %d", e)
            model.train()

            total_loss = 0
            bar: tqdm[DataLoader[PagesDataset]] = tqdm(
                dataloader, desc=f"Epoch {e}", unit="batch"
            )

            for r, c in bar:
                ruled: torch.Tensor = r.to(DEVICE)
                clean: torch.Tensor = c.to(DEVICE)

                optim.zero_grad()

                with autocast(DEVICE):
                    pred = model(ruled)
                    loss = criterion(pred, clean)

                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                scaler.step(optim)
                scaler.update()

                total_loss += loss.item()
                bar.set_postfix(loss=f"{loss.item():.5f}")
                scheduler.step()

            logger.info(
                f"Epoch {e} complete, saving, avg_loss={total_loss / len(dataloader):.5f}"
            )
            if total_loss / len(dataloader) < best_loss:
                logger.info("Best loss recorded !")
                best_loss = total_loss / len(dataloader)
            save_model(
                ModelStats(
                    epoch=e,
                    loss=total_loss / len(dataloader),
                    last_epoch_train_time=(time.time_ns() - start) / 1_000_000,
                ),
                model,
                optim,
                scheduler,
            )
