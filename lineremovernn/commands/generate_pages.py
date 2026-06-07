from argparse import Namespace
from pathlib import Path

from lineremovernn.commands.command import Command
from lineremovernn.data.iam import IAMDataset
from lineremovernn.data.mathwriting import MathWritingDataset
from lineremovernn.data.pages_generator import generate
from lineremovernn.utils import logging

logger = logging.get_logger("PageGenerator")


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
            "-j",
            "--save-json",
            action="store_true",
            help="Export ground-truth word layout coordinates as JSON files",
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

    def execute(self, args: Namespace) -> None:
        generate(
            n=args.n,
            use_arc=args.arc,
            imperfect_lines=args.imperfect_lines,
            max_warp=args.max_warp,
            save_json=args.save_json,
            seed=args.seed,
            target=args.out,
            workers=args.workers,
            preload=args.preload,
            datasets={0.51: IAMDataset, 0.49: MathWritingDataset},
        )
