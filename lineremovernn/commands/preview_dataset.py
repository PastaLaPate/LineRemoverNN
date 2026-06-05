import random
from argparse import Namespace

import matplotlib.pyplot as plt

from lineremovernn.commands.command import Command
from lineremovernn.data.pages import PagesDataset
from lineremovernn.utils import logging

logger = logging.get_logger("DatasetPreviewer")


class PreviewDatasetCommand(Command):
    def __init__(self):
        super().__init__(
            name="preview-dataset",
            description="Preview a dataset",
        )

    def init_parser(self, parser):
        parser.add_argument(
            "-d",
            "--dataset",
            type=str,
            choices=["pages"],
            default="pages",
            help="Which dataset to download (default: pages).",
        )
        parser.add_argument(
            "-n",
            "--n",
            type=int,
            default=5,
            help="How many pages to show.",
        )

    def execute(self, args: Namespace) -> None:
        if args.dataset == "pages":
            pages = PagesDataset(None)
            if len(pages) < args.n:
                raise Exception("Not enough pages in the dataset.")

            indices = random.sample(range(len(pages)), min(args.n, len(pages)))

            fig, axes = plt.subplots(2, args.n, figsize=(5 * args.n, 12))
            if args.n == 1:
                axes = [axes]

            for row, idx in enumerate(indices):
                blank, ruled = pages[idx]
                # decode_image returns CHW uint8 tensor, squeeze channel for grayscale
                axes[0][row].imshow(blank.squeeze().numpy(), cmap="gray")
                axes[1][row].imshow(ruled.squeeze().numpy(), cmap="gray")
                axes[0][row].axis("off")
                axes[1][row].axis("off")

            plt.tight_layout()
            plt.show()
