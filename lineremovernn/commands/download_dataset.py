from argparse import Namespace
from pathlib import Path

from lineremovernn.commands.command import Command
from lineremovernn.data import downloadable_datasets
from lineremovernn.data.iam import IAMDataset
from lineremovernn.data.mathwriting import MathWritingDataset
from lineremovernn.utils import logging

logger = logging.get_logger("DatasetDownloader")


class DownloadDatasetCommand(Command):
    def __init__(self):
        super().__init__(
            name="download-dataset",
            description="Download the IAM dataset for generating training data.",
        )

    def init_parser(self, parser):
        parser.add_argument(
            "-o",
            "--output-dir",
            type=Path,
            required=False,
            default=None,
            help="Directory to save the downloaded dataset.",
        )
        parser.add_argument(
            "-fd",
            "--force-download",
            action="store_true",
            help="Force re-download even if the dataset already exists.",
        )
        parser.add_argument(
            "-fe",
            "--force-extract",
            action="store_true",
            help="Force re-extract even if the dataset has been extracted.",
        )
        parser.add_argument(
            "-d",
            "--dataset",
            type=str,
            choices=[x.lower() for x in downloadable_datasets.keys()],
            default="iam",
            help="Which dataset to download (default: iam).",
        )

    def execute(self, args: Namespace) -> None:
        if args.dataset == "iam":
            dataset = IAMDataset()
            dataset.install(args.force_download, args.output_dir, args.output_dir)
        elif args.dataset == "mathwriting":
            dataset = MathWritingDataset()
            dataset.install(
                args.force_download,
                args.force_extract,
                args.output_dir,
                args.output_dir,
            )
