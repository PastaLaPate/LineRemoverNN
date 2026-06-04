from argparse import Namespace
from pathlib import Path

from lineremovernn.commands.command import Command
from lineremovernn.data.generation.pages_generator import generate
from lineremovernn.utils import logging

logger = logging.get_logger("DatasetDownloader")


class GeneratePagesCommand(Command):
    def __init__(self):
        super().__init__(
            name="generate-pages",
            description="Generate pages from the IAM dataset for training.",
        )

    def init_parser(self, parser):
        parser.add_argument(
            "-n", "--n", type=int, default=50, help="Number of page pairs to generate"
        )
        parser.add_argument(
            "-a", "--arc", action="store_true", help="Use slightly arced ruled lines"
        )
        parser.add_argument(
            "-s", "--seed", type=int, default=None, help="RNG seed for reproducibility"
        )
        parser.add_argument(
            "-i", "--iam", type=Path, default=None, help="Override IAM dataset path"
        )
        parser.add_argument(
            "-o", "--out", type=Path, default=None, help="Override output target path"
        )
        parser.add_argument(
            "-w",
            "--workers",
            type=int,
            default=None,
            help="CPU worker processes (default: all cores)",
        )
        parser.add_argument(
            "-iw",
            "--io-workers",
            type=int,
            default=16,
            help="I/O threads for image preloading (default: 16)",
        )

    def execute(self, args: Namespace) -> None:
        generate(
            n=args.n,
            use_arc=args.arc,
            seed=args.seed,
            iam_path=args.iam,
            target=args.out,
            workers=args.workers,
            io_workers=args.io_workers,
        )
