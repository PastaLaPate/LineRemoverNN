import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch

from lineremovernn.model.lineremover import LineRemovalUNet
from lineremovernn.utils.consts import DEFAULT_MODELS, DEVICE

logger = logging.getLogger("ModelLoader")


@dataclass
class ModelStats:
    epoch: int
    loss: float
    last_epoch_train_time: float


@dataclass
class SavedModel:
    stats: ModelStats
    model_state: dict
    optim_state: dict
    scheduler_state: dict


def save_model(
    stats: ModelStats,
    model: LineRemovalUNet,
    optim: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    dir: Path = DEFAULT_MODELS,
):
    path = (
        dir / f"epoch_{stats.epoch}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt"
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": stats.epoch,
            "loss": stats.loss,
            "last_epoch_train_time": stats.last_epoch_train_time,
            "model_state": model.state_dict(),
            "optimizer_state": optim.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        },
        path,
    )
    return path


def ls_models(
    dir: Path = DEFAULT_MODELS, training=True
) -> list[tuple[ModelStats, Path]]:
    models = []
    dir.mkdir(parents=True, exist_ok=True)
    for path in dir.iterdir():
        if not path.is_file() or path.suffix != ".pt":
            continue
        try:
            data = torch.load(path, map_location=DEVICE, weights_only=not training)
            models.append(
                (
                    ModelStats(
                        epoch=data["epoch"],
                        loss=data["loss"],
                        last_epoch_train_time=data["last_epoch_train_time"],
                    ),
                    path,
                )
            )
        except Exception as e:
            logger.error("Failed to load model from %s: %s", path, e)

    models.sort(key=lambda x: x[0].epoch)
    return models


def get_latest_model() -> tuple[ModelStats, Path] | None:
    models = ls_models()
    return models[-1] if models else None


def load_model(path: Path, training=True) -> SavedModel:
    data = torch.load(path, map_location=DEVICE, weights_only=not training)
    return SavedModel(
        ModelStats(
            epoch=data["epoch"],
            loss=data["loss"],
            last_epoch_train_time=data["last_epoch_train_time"],
        ),
        model_state=data["model_state"],
        optim_state=data["optimizer_state"],
        scheduler_state=data["scheduler_state"],
    )
