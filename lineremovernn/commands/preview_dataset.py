import random
from argparse import Namespace

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.v2 as v2

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
        parser.add_argument(
            "-t",
            "--transform",
            action="store_true",
            help="Add some random transforms",
        )

    def execute(self, args: Namespace) -> None:
        if args.dataset == "pages":
            pages = PagesDataset(
                v2.Compose(
                    [
                        v2.RandomCrop(
                            (384, 384),
                            pad_if_needed=True,
                            fill=0,
                            padding_mode="constant",
                        ),
                        v2.RandomPerspective(distortion_scale=0.15, p=0.5, fill=0),
                        v2.RandomAffine(
                            degrees=(-1.5, 1.5),
                            scale=(
                                0.75,
                                1.3,
                            ),
                            shear=(-4, 4),
                            fill=0,
                        ),
                        v2.RandomCrop(
                            (256, 256),
                            pad_if_needed=True,
                            fill=0,
                            padding_mode="constant",
                        ),
                        v2.ToDtype(torch.float32, scale=True),
                    ]
                )
                if args.transform
                else None,
                True,
            )
            if len(pages) < args.n:
                raise Exception("Not enough pages in the dataset.")

            indices = random.sample(range(len(pages)), min(args.n, len(pages)))
            fig, axes = plt.subplots(2, args.n, figsize=(5 * args.n, 12), squeeze=False)

            for col_idx, idx in enumerate(indices):
                blank, ruled, page = pages[idx]
                axes[0][col_idx].imshow(blank.squeeze().numpy(), cmap="gray")
                axes[1][col_idx].imshow(ruled.squeeze().numpy(), cmap="gray")

                if page and page.lines:
                    for line in page.lines_with_boxes():
                        if not line:
                            continue

                        xs1, ys1, xs2, ys2 = [], [], [], []
                        for _word, box in line:
                            x, y, w, h = box.tolist()
                            xs1.append(x)
                            ys1.append(y)
                            xs2.append(x + w)
                            ys2.append(y + h)
                        line_x, line_y = min(xs1), min(ys1)
                        line_w = max(xs2) - line_x
                        line_h = max(ys2) - line_y

                        for ax in (axes[0][col_idx], axes[1][col_idx]):
                            ax.add_patch(
                                patches.Rectangle(
                                    (line_x, line_y),
                                    line_w,
                                    line_h,
                                    linewidth=1.5,
                                    edgecolor="lime",
                                    facecolor="none",
                                )
                            )

                        for word, box in line:
                            x, y, w, h = box.tolist()

                            for ax in (axes[0][col_idx], axes[1][col_idx]):
                                ax.add_patch(
                                    patches.Rectangle(
                                        (x, y),
                                        w,
                                        h,
                                        linewidth=1,
                                        edgecolor="red",
                                        facecolor="none",
                                    )
                                )

                            transcript = word.transcript
                            display_text = (
                                transcript[:17] + "..."
                                if len(transcript) > 17
                                else transcript
                            )
                            for ax in (axes[0][col_idx], axes[1][col_idx]):
                                ax.text(
                                    x,
                                    y - 4,
                                    display_text,
                                    color="blue",
                                    fontsize=7,
                                    bbox=dict(
                                        facecolor="white",
                                        alpha=0.6,
                                        pad=0.5,
                                        edgecolor="none",
                                    ),
                                )

                axes[0][col_idx].axis("off")
                axes[1][col_idx].axis("off")

            plt.tight_layout()
            plt.show()
