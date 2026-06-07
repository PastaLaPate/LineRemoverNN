import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import tqdm

from lineremovernn.data.dataset import DownloadableDataset, ImageDataset
from lineremovernn.utils import logging

logger = logging.get_logger("IAM")


class IAMDataset(DownloadableDataset, ImageDataset):
    ID = "IAM"

    def __init__(self):
        super().__init__()
        self.len = 0  # Placeholder, as we don't have a specific length until we load the dataset.

    def __len__(self):
        return self.len

    def load(self):
        return super().load()

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
        # Ensure the target directory exists
        extract_to.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=True) as tmp_file:
            logger.info("Connecting to server...")

            with urlopen(url) as response:
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
                            break
                        tmp_file.write(chunk)
                        pbar.update(len(chunk))

            # Force any buffered data to write to disk and reset file pointer to the beginning
            tmp_file.flush()
            tmp_file.seek(0)

            logger.info(f"Extracting archive to {extract_to}...")
            with ZipFile(tmp_file) as zf:
                zf.extractall(path=extract_to)

        logger.info("Extraction complete and temporary files cleaned up.")
