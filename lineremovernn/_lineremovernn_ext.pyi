"""High-performance page generation and line removal (OpenCV + Cairo)"""

from collections.abc import Sequence
import enum
import os
import pathlib


class Dataset:
    def __init__(self, id: str, path: str, proportion: float = 1.0, preload: bool = False, index: bool = False) -> None: ...

    @property
    def id(self) -> str: ...

    @id.setter
    def id(self, arg: str, /) -> None: ...

    @property
    def path(self) -> pathlib.Path: ...

    @path.setter
    def path(self, arg: str | os.PathLike, /) -> None: ...

    @property
    def proportion(self) -> float: ...

    @proportion.setter
    def proportion(self, arg: float, /) -> None: ...

class BLOCK_TYPES(enum.Enum):
    TITLE = 0

    CAT_TITLE = 1

    PARAGRAPH = 2

    SCHEMA = 3

    SKIP_LINE = 4

TITLE: BLOCK_TYPES = BLOCK_TYPES.TITLE

CAT_TITLE: BLOCK_TYPES = BLOCK_TYPES.CAT_TITLE

PARAGRAPH: BLOCK_TYPES = BLOCK_TYPES.PARAGRAPH

SCHEMA: BLOCK_TYPES = BLOCK_TYPES.SCHEMA

SKIP_LINE: BLOCK_TYPES = BLOCK_TYPES.SKIP_LINE

class PageSettings:
    def __init__(self, document: bool, save_labels: bool, w: int, h: int, line_height: int, brightness: int, max_warp: float, imperfect_lines: bool, arc: bool) -> None: ...

    @property
    def document(self) -> bool: ...

    @property
    def save_labels(self) -> bool: ...

    @property
    def w(self) -> int: ...

    @property
    def h(self) -> int: ...

    @property
    def line_height(self) -> int: ...

    @property
    def brightness(self) -> int: ...

    @property
    def max_warp(self) -> float: ...

    @property
    def imperfect_lines(self) -> bool: ...

    @property
    def arc(self) -> bool: ...

class PageAsset:
    def __init__(self, dataset_id: str, idx: int, page_idx: int, w: int, h: int, x: int, y: int, scale: float, transcript: str) -> None: ...

    @property
    def dataset_id(self) -> str: ...

    @property
    def idx(self) -> int: ...

    @property
    def page_idx(self) -> int: ...

    @property
    def w(self) -> int: ...

    @property
    def h(self) -> int: ...

    @property
    def x(self) -> int: ...

    @property
    def y(self) -> int: ...

    @property
    def scale(self) -> float: ...

    @property
    def transcript(self) -> str: ...

class LayoutBlock:
    def __init__(self, type: BLOCK_TYPES, y_start: int, height: int, n_lines: int, schema_x_offset: float, line_skipped: int, assets: Sequence[Sequence[PageAsset]]) -> None: ...

    @property
    def type(self) -> BLOCK_TYPES: ...

    @property
    def y_start(self) -> int: ...

    @property
    def height(self) -> int: ...

    @property
    def n_lines(self) -> int: ...

    @property
    def schema_x_offset(self) -> float: ...

    @property
    def line_skipped(self) -> int: ...

    @property
    def assets(self) -> list[list[PageAsset]]: ...

def generate_pages(target: str | os.PathLike, datasets: Sequence[Dataset], n: int = 5, use_arc: bool = True, document: bool = True, max_warp: float = 0.1, imperfect_lines: bool = True, save_xml: bool = False, debug: bool = False, max_workers: int = 0) -> None: ...
