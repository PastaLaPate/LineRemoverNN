from io import BytesIO
from pathlib import Path
import tarfile
from urllib.request import urlopen
from zipfile import ZipFile

import tqdm

from lineremovernn.data.dataset import DownloadableDataset
from lineremovernn.utils import logging


logger = logging.get_logger("IAM")


class IAMDataset(DownloadableDataset):
    ID = "IAM"

    def __init__(self):
        super().__init__()

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return None

    @classmethod
    def download(cls, download_path: str, force: bool = False):
        dataset_path = Path(download_path) / "IAM_Words"

        if not (dataset_path / "words.tgz").exists() or force:
            logger.info("Downloading IAM dataset...")
            cls._download_and_unzip("https://git.io/J0fjL", dataset_path)
        else:
            raise FileExistsError(
                f"Dataset already exists at {dataset_path}. Use force=True to re-download."
            )

    @classmethod
    def extract(cls, download_path: str, dataset_path: str, force: bool = False):
        dl = Path(download_path) / "IAM_Words" / "IAM_Words"
        logger.info("Extracting words.tgz...")
        if not (Path(dataset_path) / "words").exists() or force:
            with tarfile.open(dl / "words.tgz") as f:
                f.extractall(Path(dataset_path) / "words")
        else:
            raise FileExistsError(
                f"Extracted dataset already exists at {Path(dataset_path) / 'words'}. Use force=True to re-extract."
            )

        logger.info("Moving the words.txt file to the dataset root...")
        if not (Path(dataset_path) / "words.txt").exists() or force:
            (Path(dl) / "words.txt").rename(Path(dataset_path) / "words.txt")
        else:
            raise FileExistsError(
                f"words.txt already exists at {Path(dataset_path) / 'words.txt'}. Use force=True to re-move."
            )

        logger.info("Done.")

    @classmethod
    def _download_and_unzip(
        cls, url: str, extract_to: Path, chunk_size: int = 1024 * 1024
    ) -> None:
        with urlopen(url) as response:
            total = response.length // chunk_size + 1
            data = b""
            for _ in tqdm.tqdm(range(total), desc="Downloading", unit="chunk"):
                data += response.read(chunk_size)

        with ZipFile(BytesIO(data)) as zf:
            zf.extractall(path=extract_to)
