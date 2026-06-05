from lineremovernn.commands.download_dataset import DownloadDatasetCommand
from lineremovernn.commands.generate_pages import GeneratePagesCommand
from lineremovernn.commands.model_info import ModelInfoCommand
from lineremovernn.commands.preview_dataset import PreviewDatasetCommand

from .command import Command

commands: list[Command] = [
    DownloadDatasetCommand(),
    GeneratePagesCommand(),
    PreviewDatasetCommand(),
    ModelInfoCommand(),
]

__all__ = []
