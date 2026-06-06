import argparse
from abc import ABC, abstractmethod


class Command(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.desc = description

    def add_command(
        self, subparser: argparse._SubParsersAction[argparse.ArgumentParser]
    ) -> None:
        parser = subparser.add_parser(self.name, help=self.desc, description=self.desc)
        parser.set_defaults(func=self.execute)
        self.init_parser(parser)

    @abstractmethod
    def init_parser(self, parser: argparse.ArgumentParser) -> None:
        pass

    @abstractmethod
    def execute(self, args: argparse.Namespace) -> None:
        pass
