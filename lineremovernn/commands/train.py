from argparse import Namespace
from pathlib import Path

from torch.export.pt2_archive.constants import MODELS_DIR

from lineremovernn.commands.command import Command
from lineremovernn.data.pages import PagesDataset
from lineremovernn.utils import logging

logger = logging.get_logger("ModelTrainer")


class ModelInfoCommand(Command):
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
            default=MODELS_DIR,
            help="Output location.",
        )

    def execute(self, args: Namespace) -> None:
        # model = LineRemovalUNet()
        pass
