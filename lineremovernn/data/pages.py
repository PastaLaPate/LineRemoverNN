import os
from typing import Tuple

from torch import Tensor
from torchvision import tv_tensors
from torchvision.io import ImageReadMode, decode_image

from lineremovernn.data.dataset import TorchDataset


class PagesDataset(TorchDataset):
    ID = "pages"

    """
    Folders must have files named 0.jpg, 1.jpg, ..., n.jpg
    """

    def __init__(self, transform=None):
        self.ruled_path = self.path() / "ruled-pages"
        self.clean_path = self.path() / "clean-pages"
        self.transform = transform

        # Check dataset validity
        if not len(os.listdir(self.ruled_path)) == len(os.listdir(self.clean_path)):
            raise ValueError(
                "Clean and ruled input directories must have the same number of files."
            )

    def __len__(self) -> int:
        return len(os.listdir(self.ruled_path))

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        ruled_img_path = self.ruled_path / f"{idx}.jpg"
        clean_img_path = self.clean_path / f"{idx}.jpg"
        ruled = decode_image(str(ruled_img_path), ImageReadMode.GRAY)
        clean = decode_image(str(clean_img_path), ImageReadMode.GRAY)
        ruled = tv_tensors.Image(ruled)
        clean = tv_tensors.Image(clean)

        if self.transform:
            ruled, clean = self.transform(ruled, clean)

        return (ruled, clean)
