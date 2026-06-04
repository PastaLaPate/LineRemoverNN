from lineremovernn.commands.download_dataset import DownloadDatasetCommand
from .command import Command

commands: list[Command] = [DownloadDatasetCommand()]

__all__ = []
