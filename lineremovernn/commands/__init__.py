from lineremovernn.commands.cpp_generate_pages import GeneratePagesCPPCommand
from lineremovernn.commands.download_dataset import DownloadDatasetCommand
from lineremovernn.commands.generate_pages import GeneratePagesCommand
from lineremovernn.commands.gui import GUIInferCommand
from lineremovernn.commands.list_models import ListModelsCommand
from lineremovernn.commands.model_info import ModelInfoCommand
from lineremovernn.commands.preview_dataset import PreviewDatasetCommand
from lineremovernn.commands.test import TestCommand
from lineremovernn.commands.train import TrainCommand

from .command import Command

commands: list[Command] = [
    DownloadDatasetCommand(),
    GeneratePagesCommand(),
    PreviewDatasetCommand(),
    ModelInfoCommand(),
    TrainCommand(),
    ListModelsCommand(),
    TestCommand(),
    GUIInferCommand(),
    GeneratePagesCPPCommand(),
]

__all__ = []
