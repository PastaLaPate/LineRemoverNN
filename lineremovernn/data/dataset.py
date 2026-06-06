import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset as TorchADataset

from lineremovernn.utils.consts import DEFAULT_DATASETS, DEFAULT_DOWNLOADS

"""
Dataset -> Standard class, name, path, etc.
  DownloadableDataset -> Dataset with download method, to download and extract. Ex: IAM, MNIST
  TorchDataset -> Dataset that can be used with PyTorch DataLoader. Ex: PagesDataset
"""

logger = logging.getLogger("Dataset")


class Dataset(ABC):
    ID = "DUMMY"

    def __init__(self):
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx) -> Any:
        pass

    @classmethod
    def path(cls) -> Path:
        return DEFAULT_DATASETS / cls.ID


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
