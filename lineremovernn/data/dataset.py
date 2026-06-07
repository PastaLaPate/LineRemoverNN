import logging
from abc import ABC, abstractmethod
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, NamedTuple

from PIL import Image
from torch.utils.data import Dataset as TorchADataset

from lineremovernn.utils.consts import DEFAULT_DATASETS, DEFAULT_DOWNLOADS

"""
Dataset -> Standard class, name, path, etc.
  DownloadableDataset -> Dataset with download method, to download and extract. Ex: IAM, MNIST
  TorchDataset -> Dataset that can be used with PyTorch DataLoader. Ex: PagesDataset
"""

logger = logging.getLogger("Dataset")


class CropAsset(NamedTuple):
    path: str
    text: str
    raw_bytes: bytes | None = None


class Dataset(ABC):
    ID = "DUMMY"

    def __init__(self):
        pass

    def available(self) -> bool:
        return self.path().exists()

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx) -> Any:
        pass

    @classmethod
    def path(cls) -> Path:
        return DEFAULT_DATASETS / cls.ID


class CachedDataset(Dataset):
    def __init__(self, preload: bool = False):
        self.preload_enabled = preload
        self.assets: list[CropAsset] = []
        self._static_cache: dict[str, bytes] = {}

        self._bounded_load = lru_cache(maxsize=3000)(self._read_raw_bytes)

    def _read_raw_bytes(self, path: str) -> bytes | None:
        try:
            with open(path, "rb") as f:
                return f.read()
        except Exception:
            return None

    @abstractmethod
    def load(self):
        """Loads the dataset metadata (e.g., file paths, labels) into memory, but not the raw bytes. Populates self.assets."""
        pass

    def preload(self) -> None:
        """Preloads all unique images directly into this instance's static cache storage."""
        unique_paths = list({asset.path for asset in self.assets})
        loaded = 0
        for p in unique_paths:
            data = self._read_raw_bytes(p)
            if data is not None:
                self._static_cache[p] = data
                loaded += 1
        logging.getLogger("Dataset").info(
            f"[{self.ID}] Preloaded {loaded} assets into instance cache."
        )

    def __getitem__(self, idx: int) -> CropAsset:
        meta = self.assets[idx]

        if self.preload_enabled:
            # Look up instantly from our preloaded dictionary
            raw = self._static_cache.get(meta.path)
        else:
            # Read through our worker-local instance LRU cache layer
            raw = self._bounded_load(meta.path)

        return CropAsset(path=meta.path, text=meta.text, raw_bytes=raw)


class ImageDataset(CachedDataset):
    def get_image(self, idx: int) -> Image.Image:
        asset = super().__getitem__(idx)
        if asset.raw_bytes is None:
            raise FileNotFoundError(f"Could not read image bytes from {asset.path}")
        return Image.open(BytesIO(asset.raw_bytes)).convert("RGBA")


class DownloadableDataset(Dataset):
    @classmethod
    def download_path(cls) -> Path:
        return DEFAULT_DOWNLOADS / cls.ID

    @classmethod
    def install(
        cls,
        force_download: bool = False,
        force_extract: bool = False,
        download_path: str | None = None,
        dataset_path: str | None = None,
    ):
        if download_path is None:
            download_path = str(cls.download_path())
        if dataset_path is None:
            dataset_path = str(cls.path())

        if not Path(download_path).exists() or force_download:
            cls.download(force=force_download, download_path=download_path)
        else:
            logger.warning(
                f"{cls.ID} already downloaded at {download_path}, use force download to re download it."
            )

        if not Path(dataset_path).exists() or force_extract:
            cls.extract(
                force=force_extract,
                download_path=download_path,
                dataset_path=dataset_path,
            )
        else:
            logger.warning(
                f"{cls.ID} already extracted at {dataset_path}, use force extract to re extract it."
            )

    @classmethod
    @abstractmethod
    def download(cls, download_path: str, force: bool = False):
        pass

    @classmethod
    @abstractmethod
    def extract(cls, download_path: str, dataset_path: str, force: bool = False):
        pass


class TorchDataset(Dataset, TorchADataset):
    pass
