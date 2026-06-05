from argparse import Namespace

from torchinfo import summary

from lineremovernn.commands.command import Command
from lineremovernn.model.model import LineRemovalUNet
from lineremovernn.utils import logging

logger = logging.get_logger("ModelInfo")


class ModelInfoCommand(Command):
    def __init__(self):
        super().__init__(
            name="model-info",
            description="Preview the model's architecture",
        )

    def init_parser(self, parser):
        pass

    def execute(self, args: Namespace) -> None:
        model = LineRemovalUNet()
        summary(model, input_size=(8, 1, 256, 256))
