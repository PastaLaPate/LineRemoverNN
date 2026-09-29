import argparse
from argparse import Namespace
from collections.abc import Sequence
from typing import ClassVar

from lineremovernn._lineremovernn_ext import Dataset, generate_pages
from lineremovernn.commands.command import Command
from lineremovernn.data.ai2d import AI2DDataset
from lineremovernn.data.iam import IAMDataset
from lineremovernn.data.mathwriting import MathWritingDataset
from lineremovernn.data.pages import PagesDataset
from lineremovernn.utils import logging

logger = logging.get_logger("PageGenerator")

# Ex uv run lineremovernn generate-pages -m -il -a -n 15 --datasets iam:1:ip mathwriting:0.3


class ParseDatasets(argparse.Action):
    ALLOWED_DATASETS: ClassVar[set[str]] = {
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
        datasets_dict: dict[str, tuple[float, bool, bool]] = {}

        if not values:
            raise parser.error("No datasets given")
        if isinstance(values, str):
            values = [values]

        for item in values:
            try:
                split_ = item.split(":")
                if len(split_) == 2:
                    name, proportion_str = split_
                    flags_str = ""
                elif len(split_) == 3:
                    name, proportion_str, flags_str = split_
                proportion = float(proportion_str)
                preload = "p" in flags_str
                index = "i" in flags_str
            except ValueError:
                raise parser.error(
                    f"Invalid format for '{item}'. Must be 'name:proportion:flags' (e.g., mathwriting:0.5, iam:1:ip)"
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

            datasets_dict[name_lower] = (proportion, preload, index)
        datasets: list[Dataset] = []
        for dataset_id, properties in datasets_dict.items():
            if dataset_id == IAMDataset.ID.lower():
                if not IAMDataset.available():
                    raise parser.error("IAM Dataset isn't available")
                datasets.append(
                    Dataset(
                        IAMDataset.ID.lower(),
                        str(IAMDataset.path()),
                        *properties,
                    )
                )
            elif dataset_id == MathWritingDataset.ID.lower():
                if not MathWritingDataset.available():
                    raise parser.error("Mathwriting Dataset isn't available")
                datasets.append(
                    Dataset(
                        MathWritingDataset.ID.lower(),
                        str(MathWritingDataset.path()),
                        *properties,
                    )
                )
            elif dataset_id == AI2DDataset.ID.lower():
                if not AI2DDataset.available():
                    raise parser.error("AI2D Dataset isn't available")
                datasets.append(
                    Dataset(
                        AI2DDataset.ID.lower(),
                        str(AI2DDataset.path()),
                        *properties,
                    )
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
            default=[
                Dataset(IAMDataset.ID.lower(), str(IAMDataset.path()), 1)
            ],
            help="Space-separated datasets, proportions and flags: p for preload and i for indexing (e.g., iam:1:ip mathwriting:0.3)",
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
        parser.add_argument(
            "-db",
            "--debug",
            action="store_true",
            default=None,
            help="Debug generation speed.",
        )

    def execute(self, args: Namespace) -> None:
        generate_pages(
            PagesDataset.path(),
            datasets=args.datasets,
            n=args.n,
            use_arc=args.arc,
            max_warp=args.max_warp,
            imperfect_lines=args.imperfect_lines,
            save_xml=args.save_metadata,
            document=args.docs,
            debug=args.debug or False,
            max_workers=args.workers or 0,
            logger=logger,
        )
