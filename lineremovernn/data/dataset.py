from abc import ABC, abstractmethod
from pathlib import Path
from torch.utils.data import Dataset as TorchADataset
from typing import Any

from lineremovernn.utils.consts import DEFAULT_DATASETS, DEFAULT_DOWNLOADS

"""
Dataset -> Standard class, name, path, etc.
  DownloadableDataset -> Dataset with download method, to download and extract. Ex: IAM, MNIST
  TorchDataset -> Dataset that can be used with PyTorch DataLoader. Ex: PagesDataset
"""

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
    def install(cls, force: bool = False, download_path: str | None = None, dataset_path: str | None = None):
        if download_path is None:
            download_path = str(cls.download_path())
        if dataset_path is None:
            dataset_path = str(cls.path())

        if not Path(download_path).exists() or force:
            cls.download(force=force, download_path=download_path)
        else:
            print(f"{cls.ID} already downloaded at {download_path}")
            
        if not Path(dataset_path).exists() or force:
            cls.extract(force=force, download_path=download_path, dataset_path=dataset_path)
        else:
            print(f"{cls.ID} already extracted at {dataset_path}")
        

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