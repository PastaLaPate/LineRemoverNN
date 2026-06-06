import argparse

from torch.cuda import is_available as torch_cuda_available

import lineremovernn.utils.logging as logging
from lineremovernn import commands
from lineremovernn.utils.consts import ARCH, OS, PYTHON_VERSION, VERSION

logger = logging.get_logger("Main")


def main():
    logger.info("Running on LineRemoverNN version %s", VERSION)
    logger.debug("OS: %s | Python: %s | Architecture: %s", OS, PYTHON_VERSION, ARCH)
    logger.info(
        "Cuda availability : %s",
        "available" if torch_cuda_available() else "not available",
    )

    parser = argparse.ArgumentParser(
        description="LineRemoverNN: A tool for removing lines from handwritten text images using neural networks."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in commands.commands:
        command.add_command(subparsers)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
