from argparse import Namespace

from lineremovernn.commands.command import Command
from lineremovernn.utils import logging
from lineremovernn.utils.saver import ls_models

logger = logging.get_logger("ListModels")


def format_human_time(delta: int) -> str:
    total_seconds = delta
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours} {'Hour' if hours == 1 else 'Hours'}")
    if minutes > 0 or hours > 0:  # Show minutes if hours exist, even if 0
        parts.append(f"{minutes} {'Minute' if minutes == 1 else 'Minutes'}")
    parts.append(f"{seconds} {'Second' if seconds == 1 else 'Seconds'}")

    # Join with commas, and use "and" for the last element
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


class ListModelsCommand(Command):
    def __init__(self):
        super().__init__(
            name="ls-models",
            description="List the available models",
        )

    def init_parser(self, parser):
        pass

    def execute(self, args: Namespace) -> None:
        models = ls_models()
        print("#################################")
        for stats, path in models:
            print(f"Filename : {path.name}")
            print(f"Epoch : {stats.epoch}")
            print(f"Avg loss : {stats.loss}")
            print(
                f"Train time : {format_human_time(int(stats.last_epoch_train_time / 1000))}"
            )
            print("#################################")
