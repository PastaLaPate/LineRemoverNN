import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import tqdm
from PIL import Image

from lineremovernn.data.dataset import DownloadableDataset, ImageDataset
from lineremovernn.utils import logging

logger = logging.get_logger("AI2D")


class AI2DDataset(DownloadableDataset, ImageDataset):
    ID = "AI2D"
    DOWNLOAD_URL = "http://ai2-website.s3.amazonaws.com/data/ai2d-all.zip"
    FILENAME = "ai2d-all.zip"

    def __init__(self, preload: bool = False):
        DownloadableDataset.__init__(self)
        ImageDataset.__init__(self, preload=preload)

    def _load_metadata(self):
        return super()._load_metadata()

    def get_image(self, idx: int, mode="RGBA") -> Image.Image:
        return Image.fromarray(np.zeros((1, 255, 255)))

    @classmethod
    def download(cls, download_path: str, force: bool = False):
        download_p = Path(download_path)
        download_p.mkdir(parents=True, exist_ok=True)
        target_file = download_p / cls.FILENAME

        if not target_file.exists() or force:
            logger.info(f"Downloading AI2D dataset from source: {cls.DOWNLOAD_URL}")
            chunk_size = 1024 * 1024  # 1MB chunks

            with (
                urlopen(cls.DOWNLOAD_URL) as response,
                open(target_file, "wb") as f,
            ):
                total_size = int(response.headers.get("Content-Length", 0))

                with tqdm.tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc="Downloading AI2D",
                    leave=True,
                ) as pbar:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))
        else:
            raise FileExistsError(
                f"Dataset archive already exists at {target_file}. Use force=True to force installation modifications."
            )

    @classmethod
    def extract(cls, download_path: str, dataset_path: str, force: bool = False):
        download_p = Path(download_path)
        dataset_p = Path(dataset_path) / "ai2d"
        archive_source = download_p / cls.FILENAME

        logger.info(f"Extracting package contents from {cls.FILENAME}...")
        if not (dataset_p).exists() or force:
            with zipfile.ZipFile(archive_source, "r") as zip_ref:
                file_list = zip_ref.infolist()

                with tqdm.tqdm(
                    total=len(file_list),
                    unit="file",
                    desc="Extracting AI2D",
                    leave=True,
                ) as pbar:
                    for file in file_list:
                        zip_ref.extract(member=file, path=dataset_p)
                        pbar.update(1)
        else:
            raise FileExistsError(
                f"Extracted destination files already exist at {dataset_p}."
            )

        logger.info("AI2D environment initialization finalized.")
