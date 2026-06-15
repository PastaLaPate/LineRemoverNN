"""High-performance page generation and line removal (OpenCV + Cairo)"""

import os
import pathlib
from collections.abc import Sequence

class Dataset:
    def __init__(self, id: str, path: str, proportion: float = 1.0) -> None: ...
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

def generate_pages(
    target: str | os.PathLike,
    datasets: Sequence[Dataset],
    n: int = 5,
    preload: bool = False,
    use_arc: bool = True,
    document: bool = True,
    max_warp: float = 0.1,
    imperfect_lines: bool = True,
    save_json: bool = False,
) -> None: ...
