import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Tuple

import torch
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
    transcript: str


@dataclass
class Page:
    idx: int
    w: int
    h: int
    line_height: int
    margin_left: int
    brightness: int
    lines: list[list[Word]]
    boxes: tv_tensors.BoundingBoxes

    def words_with_boxes(self):
        """Yield (line_idx, Word, box) for every word, box already split per-line."""
        flat_idx = 0
        for line_idx, line in enumerate(self.lines):
            for word in line:
                yield line_idx, word, self.boxes[flat_idx]
                flat_idx += 1

    def lines_with_boxes(self) -> list[list[Tuple["Word", Tensor]]]:
        """Same shape as `lines`, but each entry is (Word, box) instead of just Word."""
        out = []
        flat_idx = 0
        for line in self.lines:
            line_out = []
            for word in line:
                line_out.append((word, self.boxes[flat_idx]))
                flat_idx += 1
            out.append(line_out)
        return out


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
        ruled = decode_image(str(ruled_img_path), ImageReadMode.GRAY)
        clean = decode_image(str(clean_img_path), ImageReadMode.GRAY)
        ruled = tv_tensors.Image(ruled)
        clean = tv_tensors.Image(clean)

        if not self.load_label:
            if self.transform:
                ruled, clean = self.transform(ruled, clean)

            return (ruled, clean, None)
        label_path = self.labels_path / f"{idx}.xml"
        if not label_path.exists():
            logger.warning("Couldnt load ", idx, "'s metadata file.")
            if self.transform:
                ruled, clean = self.transform(ruled, clean)

            return (ruled, clean, None)
        tree = ET.parse(label_path)
        root = tree.getroot()

        lines_list: list[list[Word]] = []
        boxes_list: list[list[int]] = []  # flat, XYWH, reading order

        for line_elem in root.findall("line"):
            words_list = []
            for word_elem in line_elem.findall("word"):
                word = Word(
                    idx=int(word_elem.attrib["idx"]),
                    dataset_idx=int(word_elem.attrib["dataset_idx"]),
                    dataset=word_elem.attrib["dataset"],
                    transcript=word_elem.text if word_elem.text else "",
                )
                words_list.append(word)
                boxes_list.append(
                    [
                        int(word_elem.attrib["x"]),
                        int(word_elem.attrib["y"]),
                        int(word_elem.attrib["w"]),
                        int(word_elem.attrib["h"]),
                    ]
                )

            lines_list.append(words_list)

        page_w = int(root.attrib["w"])
        page_h = int(root.attrib["h"])

        if boxes_list:
            boxes_tensor = torch.tensor(boxes_list, dtype=torch.float32)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)

        boxes = tv_tensors.BoundingBoxes(
            boxes_tensor,
            format=tv_tensors.BoundingBoxFormat.XYWH,
            canvas_size=(page_h, page_w),
        )  # type: ignore

        if self.transform:
            # Pass boxes through the same call so torchvision's v2 transforms (crop/resize/flip/etc.) apply matching geometric ops to them.
            ruled, clean, boxes = self.transform(ruled, clean, boxes)

        page = Page(
            idx=int(root.attrib["idx"]),
            w=page_w,
            h=page_h,
            line_height=int(root.attrib["line_height"]),
            margin_left=int(root.attrib["margin_left"]),
            brightness=int(root.attrib["brightness"]),
            lines=lines_list,
            boxes=boxes,
        )

        return (ruled, clean, page)
