import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Tuple

from torch import Tensor
from torchvision import tv_tensors
from torchvision.io import ImageReadMode, decode_image

from lineremovernn.data.dataset import TorchDataset
from lineremovernn.utils import logging

logger = logging.get_logger("PagesDataset")


@dataclass
class Word:
    idx: int
    dataset_idx: int
    dataset: str
    x: int
    y: int
    w: int
    h: int
    transcript: str


@dataclass
class Page:
    idx: int
    w: int
    h: int
    line_height: int
    margin_left: int
    brightness: int
    lines: list[Word]


class PagesDataset(TorchDataset):
    ID = "pages"

    """
    Folders must have files named 0.jpg, 1.jpg, ..., n.jpg
    """

    def __init__(self, transform=None, load_label=False):
        self.ruled_path = self.path() / "ruled-pages"
        self.clean_path = self.path() / "clean-pages"
        self.labels_path = self.path() / "labels"
        self.transform = transform
        self.load_label = load_label

        # Check dataset validity
        if not len(os.listdir(self.ruled_path)) == len(os.listdir(self.clean_path)):
            raise ValueError(
                "Clean and ruled input directories must have the same number of files."
            )

    def __len__(self) -> int:
        return len(os.listdir(self.ruled_path))

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, Page | None]:
        ruled_img_path = self.ruled_path / f"{idx}.jpg"
        clean_img_path = self.clean_path / f"{idx}.jpg"
        label_path = self.labels_path / f"{idx}.xml"
        ruled = decode_image(str(ruled_img_path), ImageReadMode.GRAY)
        clean = decode_image(str(clean_img_path), ImageReadMode.GRAY)
        ruled = tv_tensors.Image(ruled)
        clean = tv_tensors.Image(clean)

        if self.transform:
            ruled, clean = self.transform(ruled, clean)
        if not self.load_label:
            return (ruled, clean, None)
        label_path = self.labels_path / f"{idx}.xml"
        if not label_path.exists():
            logger.warning("Couldnt load ", idx, "'s metadata file.")
            return (ruled, clean, None)
        tree = ET.parse(label_path)
        root = tree.getroot()

        words_list = []
        for line_elem in root.findall("line"):
            for word_elem in line_elem.findall("word"):
                word = Word(
                    idx=int(word_elem.attrib["idx"]),
                    dataset_idx=int(word_elem.attrib["dataset_idx"]),
                    dataset=word_elem.attrib["dataset"],
                    x=int(word_elem.attrib["x"]),
                    y=int(word_elem.attrib["y"]),
                    w=int(word_elem.attrib["w"]),
                    h=int(word_elem.attrib["h"]),
                    transcript=word_elem.text if word_elem.text else "",
                )
                words_list.append(word)

        page = Page(
            idx=int(root.attrib["idx"]),
            w=int(root.attrib["w"]),
            h=int(root.attrib["h"]),
            line_height=int(root.attrib["line_height"]),
            margin_left=int(root.attrib["margin_left"]),
            brightness=int(root.attrib["brightness"]),
            lines=words_list,
        )
        return (ruled, clean, page)
