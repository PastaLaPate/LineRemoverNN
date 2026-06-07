import shutil
import tarfile
from pathlib import Path
from urllib.request import urlopen

import tqdm

from lineremovernn.data.dataset import DownloadableDataset
from lineremovernn.utils import logging

logger = logging.get_logger("MathWriting")


class RawMathWriting(DownloadableDataset):
    ID = "MathWriting"
    EXCERPT_MODE = False
    DOWNLOAD_URL = (
        "https://storage.googleapis.com/mathwriting_data/mathwriting-2024-excerpt.tgz"
        if EXCERPT_MODE
        else "https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz"
    )
    FILENAME = (
        "mathwriting-2024-excerpt.tgz" if EXCERPT_MODE else "mathwriting-2024.tgz"
    )

    def __init__(self):
        super().__init__()

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return None

    @classmethod
    def download(cls, download_path: str, force: bool = False):
        download_p = Path(download_path)
        download_p.mkdir(parents=True, exist_ok=True)

        if not (download_p / cls.FILENAME).exists() or force:
            logger.info("Downloading MathWriting dataset...")
            chunk_size = 1024 * 1024  # 1MB chunks

            with (
                urlopen(cls.DOWNLOAD_URL) as response,
                open(download_p / cls.FILENAME, "wb") as f,
            ):
                # Get total file size from headers for an accurate progress bar
                total_size = int(response.headers.get("Content-Length", 0))

                with tqdm.tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc="Downloading",
                    leave=True,
                ) as pbar:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break  # Download complete

                        f.write(chunk)
                        pbar.update(len(chunk))

        else:
            raise FileExistsError(
                f"Dataset already exists at {download_p / cls.FILENAME}. Use force=True to re-download."
            )

    @classmethod
    def extract(cls, download_path: str, dataset_path: str, force: bool = False):
        download_p = Path(download_path)
        dataset_p = Path(dataset_path)
        logger.info(f"Extracting {cls.FILENAME}...")
        if not (dataset_p / "readme.md").exists() or force:
            with tarfile.open(download_p / cls.FILENAME) as f:
                f.extractall(dataset_p)
            nested_folder = (
                dataset_p / "mathwriting-2024-excerpt"
                if cls.EXCERPT_MODE
                else dataset_p / "mathwriting-2024"
            )
            if nested_folder.exists() and nested_folder.is_dir():
                logger.info("Flattening nested directory structure...")
                for item in nested_folder.iterdir():
                    shutil.move(str(item), str(dataset_p / item.name))

                nested_folder.rmdir()
        else:
            raise FileExistsError(
                f"Extracted dataset already exists at {Path(dataset_path) / 'readme.md'}. Use force=True to re-extract."
            )

        logger.info("Done.")
