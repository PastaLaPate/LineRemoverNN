import random
from argparse import Namespace

import matplotlib.patches as patches
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
            pages = PagesDataset(None, True)
            if len(pages) < args.n:
                raise Exception("Not enough pages in the dataset.")

            indices = random.sample(range(len(pages)), min(args.n, len(pages)))

            fig, axes = plt.subplots(2, args.n, figsize=(5 * args.n, 12), squeeze=False)

            for col_idx, idx in enumerate(indices):
                blank, ruled, page = pages[idx]

                axes[0][col_idx].imshow(blank.squeeze().numpy(), cmap="gray")
                axes[1][col_idx].imshow(ruled.squeeze().numpy(), cmap="gray")

                # If page metadata is defined, draw bounding boxes and text
                if page and page.lines:
                    for word in page.lines:
                        rect_blank = patches.Rectangle(
                            (word.x, word.y),
                            word.w,
                            word.h,
                            linewidth=1,
                            edgecolor="red",
                            facecolor="none",
                        )
                        rect_ruled = patches.Rectangle(
                            (word.x, word.y),
                            word.w,
                            word.h,
                            linewidth=1,
                            edgecolor="red",
                            facecolor="none",
                        )

                        axes[0][col_idx].add_patch(rect_blank)
                        axes[1][col_idx].add_patch(rect_ruled)

                        transcript = word.transcript
                        display_text = (
                            transcript[:17] + "..."
                            if len(transcript) > 17
                            else transcript
                        )
                        axes[0][col_idx].text(
                            word.x,
                            word.y - 4,
                            display_text,
                            color="blue",
                            fontsize=7,
                            bbox=dict(
                                facecolor="white", alpha=0.6, pad=0.5, edgecolor="none"
                            ),
                        )
                        axes[1][col_idx].text(
                            word.x,
                            word.y - 4,
                            display_text,
                            color="blue",
                            fontsize=7,
                            bbox=dict(
                                facecolor="white", alpha=0.6, pad=0.5, edgecolor="none"
                            ),
                        )

                axes[0][col_idx].axis("off")
                axes[1][col_idx].axis("off")

            plt.tight_layout()
            plt.show()
