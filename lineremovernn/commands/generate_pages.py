import argparse
from argparse import Namespace
from typing import Sequence

from lineremovernn._lineremovernn_ext import Dataset, generate_pages
from lineremovernn.commands.command import Command
from lineremovernn.data.ai2d import AI2DDataset
from lineremovernn.data.iam import IAMDataset
from lineremovernn.data.mathwriting import MathWritingDataset
from lineremovernn.data.pages import PagesDataset
from lineremovernn.utils import logging

logger = logging.get_logger("PageGenerator")

# Ex uv run lineremovernn generate-pages -m -il -a -n 15 --datasets iam:1 mathwriting:0.3


class ParseDatasets(argparse.Action):
    ALLOWED_DATASETS = {
        IAMDataset.ID.lower(),
        MathWritingDataset.ID.lower(),
        AI2DDataset.ID.lower(),
    }

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: Namespace,
        values: str | Sequence[str] | None,
        option_string: str | None = None,
    ) -> None:
        datasets_dict: dict[str, float] = {}

        if not values:
            raise parser.error("No datasets given")
        if isinstance(values, str):
            values = [values]

        for item in values:
            try:
                name, proportion_str = item.split(":")
                proportion = float(proportion_str)
            except ValueError:
                raise parser.error(
                    f"Invalid format for '{item}'. Must be 'name:proportion' (e.g., iam:0.5)"
                )
            name_lower = name.lower()
            if name_lower not in self.ALLOWED_DATASETS:
                raise parser.error(
                    f"Unknown dataset ID '{name}'. Allowed IDs are: {', '.join(sorted(self.ALLOWED_DATASETS))}"
                )

            if name_lower in datasets_dict:
                raise parser.error(
                    f"Duplicate dataset ID detected: '{name}' was provided more than once."
                )

            datasets_dict[name_lower] = proportion
        datasets: list[Dataset] = []
        for dataset_id, p in datasets_dict.items():
            if dataset_id == IAMDataset.ID.lower():
                if not IAMDataset.available():
                    raise parser.error("IAM Dataset isn't available")
                datasets.append(
                    Dataset(IAMDataset.ID.lower(), str(IAMDataset.path()), p)
                )
            elif dataset_id == MathWritingDataset.ID.lower():
                if not MathWritingDataset.available():
                    raise parser.error("Mathwriting Dataset isn't available")
                datasets.append(
                    Dataset(
                        MathWritingDataset.ID.lower(),
                        str(MathWritingDataset.path()),
                        p,
                    )
                )
            elif dataset_id == AI2DDataset.ID.lower():
                if not AI2DDataset.available():
                    raise parser.error("AI2D Dataset isn't available")
                datasets.append(
                    Dataset(AI2DDataset.ID.lower(), str(AI2DDataset.path()), p)
                )

        setattr(namespace, self.dest, datasets)


class GeneratePagesCPPCommand(Command):
    def __init__(self):
        super().__init__(
            name="generate-pages",
            description="Generate pages from the specified datasets for training.",
        )

    def init_parser(self, parser):
        parser.add_argument(
            "--datasets",
            nargs="+",
            action=ParseDatasets,
            default=[Dataset(IAMDataset.ID.lower(), str(IAMDataset.path()), 1)],
            help="Space-separated datasets and proportions (e.g., iam:1 mathwriting:0.3)",
        )
        parser.add_argument(
            "-n",
            "--n",
            type=int,
            default=50,
            help="Number of page pairs to generate",
        )
        parser.add_argument(
            "-a",
            "--arc",
            action="store_true",
            help="Use slightly arced ruled lines",
        )
        parser.add_argument(
            "-mw",
            "--max-warp",
            type=float,
            default=0.1,
            help="Maximum perspective warp factor for word crops (0.0 to disable, old default was 0.3)",
        )
        parser.add_argument(
            "-il",
            "--imperfect-lines",
            action="store_true",
            help="Inject tiny structural imperfections and gaps into rules",
        )
        parser.add_argument(
            "-p",
            "--preload",
            action="store_true",
            help="Preload the images in RAM.",
        )
        parser.add_argument(
            "-m",
            "--save-metadata",
            action="store_true",
            help="Export ground-truth word layout coordinates as XML files",
        )
        parser.add_argument(
            "-d",
            "--docs",
            action="store_true",
            help="Make documents like pages.",
        )
        parser.add_argument(
            "-w",
            "--workers",
            type=int,
            default=None,
            help="CPU worker processes (default: all cores)",
        )

    def execute(self, args: Namespace) -> None:
        generate_pages(
            PagesDataset.path(),
            datasets=args.datasets,
            n=args.n,
            preload=args.preload,
            use_arc=args.arc,
            max_warp=args.max_warp,
            imperfect_lines=args.imperfect_lines,
            save_xml=args.save_metadata,
            document=args.docs,
        )
