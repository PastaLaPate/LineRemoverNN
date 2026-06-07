import math
import random
import shutil
import tarfile
import xml.etree.ElementTree as ElementTree
from functools import lru_cache
from pathlib import Path
from urllib.request import urlopen

import cairo
import numpy as np
import tqdm
from PIL import Image

from lineremovernn.data.dataset import CropAsset, DownloadableDataset, ImageDataset
from lineremovernn.utils import logging

logger = logging.get_logger("MathWriting")


# ---------------------------------------------------------------------------
# Lightweight Inline InkML Entities
# ---------------------------------------------------------------------------


class InkMLParser:
    """Namespace for reading and parsing stroke vector points from MathWriting XML files."""

    @staticmethod
    def parse_file(filename: Path) -> tuple[list[np.ndarray], str]:
        """Reads trace vector coordinates and extracts textual annotations."""
        try:
            with open(filename, "r", encoding="UTF-8") as f:
                root = ElementTree.fromstring(f.read())
        except Exception as e:
            logger.error(f"Failed to parse InkML text structure at {filename}: {e}")
            return [], ""

        strokes = []
        label = ""

        for element in root:
            tag_name = element.tag.removeprefix("{http://www.w3.org/2003/InkML}")

            if tag_name == "annotation":
                attrib_type = element.attrib.get("type", "")
                # Prioritize normalizations or simple text translations
                if attrib_type in ("normalizedLabel", "label") and not label:
                    label = element.text or ""

            elif tag_name == "trace":
                if not element.text:
                    continue
                points = element.text.strip().split(",")
                stroke_x, stroke_y = [], []

                for point in points:
                    parts = point.strip().split(" ")
                    if len(parts) >= 2:
                        stroke_x.append(float(parts[0]))
                        stroke_y.append(float(parts[1]))

                if stroke_x:
                    strokes.append(np.array([stroke_x, stroke_y], dtype=np.float32))

        return strokes, label


# ---------------------------------------------------------------------------
# Complete Core Dataset Wrapper
# ---------------------------------------------------------------------------


class MathWritingDataset(DownloadableDataset, ImageDataset):
    ID = "MathWriting"
    EXCERPT_MODE = False
    DOWNLOAD_URL = (
        "https://storage.googleapis.com/mathwriting_data/mathwriting-2024-excerpt.tgz"
        if EXCERPT_MODE
        else "https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz"
    )
    FILENAME = (
        "mathwriting-2024-excerpt.tgz" if EXCERPT_MODE else "mathwriting-2024.tgz"
    )

    def __init__(self, preload: bool = False):
        DownloadableDataset.__init__(self)
        ImageDataset.__init__(self, preload=preload)

        if self.available():
            self._load_metadata()

    def _load_metadata(self) -> None:
        """Indexes available math formula paths."""
        train_dir = self.path() / "train"
        synth_dir = self.path() / "synthetic"

        # Scrape directories if they are unpackaged and present on disk
        targets = [train_dir, synth_dir]
        for target_path in targets:
            if target_path.exists():
                for file_p in target_path.glob("*.inkml"):
                    # Lazy instantiation: Text translation is resolved during parsing
                    self.assets.append(CropAsset(path=str(file_p), text=""))

        logger.info(
            f"MathWriting system index compiled. Registered {len(self.assets)} math expression formulas."
        )

        if self.preload_enabled:
            self.preload()

    @lru_cache(maxsize=2000)
    def get_image(self, idx: int) -> Image.Image:
        """
        Parses underlying ink strokes from the file system and performs on-the-fly
        vector-to-rasterization processing directly to transparent RGBA matrices.
        """
        # Resolves correct tracking metadata profile via base class assignment
        asset = ImageDataset.__getitem__(self, idx)
        file_path = Path(asset.path)

        # 1. Parse trace points from the file system
        strokes, transcript = InkMLParser.parse_file(file_path)

        # Retroactively cache the text label if it wasn't extracted during metadata initialization
        if transcript and not asset.text:
            self.assets[idx] = CropAsset(asset.path, transcript, asset.raw_bytes)

        if not strokes:
            # Fallback for empty/malformed vector structures
            return Image.new("RGBA", (8, 8), (0, 0, 0, 0))

        # 2. Compute extreme stroke limits for spatial box tracking
        all_mins = [stroke.min(axis=1) for stroke in strokes]
        all_maxs = [stroke.max(axis=1) for stroke in strokes]

        xmin, ymin = np.vstack(all_mins).min(axis=0)
        xmax, ymax = np.vstack(all_maxs).max(axis=0)

        margin = 6
        stroke_width = random.uniform(2.5, 7)
        pen_darkness = random.randint(50, 120)

        width = max(1, int(xmax - xmin + 2 * margin))
        height = max(1, int(ymax - ymin + 2 * margin))

        shift_x = -xmin + margin
        shift_y = -ymin + margin

        # 3. Rasterize vectors to native Cairo structures using clear transparency configurations
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        ctx = cairo.Context(surface)

        # Set transparent background color explicitly to skip subsequent pixel loops
        ctx.set_source_rgba(0.0, 0.0, 0.0, 0.0)
        ctx.set_operator(cairo.OPERATOR_SOURCE)
        ctx.paint()

        # Render stroke lines (Default color profile is dark ink)
        ctx.set_operator(cairo.OPERATOR_OVER)
        ctx.set_source_rgb(
            pen_darkness / 255.0, pen_darkness / 255.0, pen_darkness / 255.0
        )
        ctx.set_line_width(stroke_width)
        ctx.set_line_cap(cairo.LineCap.ROUND)
        ctx.set_line_join(cairo.LineJoin.ROUND)

        for stroke in strokes:
            n_points = stroke.shape[1]
            if n_points == 1:
                # Isolated punctuation/dots are drawn as uniform disks
                ctx.arc(
                    stroke[0, 0] + shift_x,
                    stroke[1, 0] + shift_y,
                    stroke_width / 2,
                    0,
                    2 * math.pi,
                )
                ctx.fill()
            else:
                ctx.move_to(stroke[0, 0] + shift_x, stroke[1, 0] + shift_y)
                for pt_idx in range(1, n_points):
                    ctx.line_to(
                        stroke[0, pt_idx] + shift_x, stroke[1, pt_idx] + shift_y
                    )
                ctx.stroke()

        # 4. Flush the byte arrays into standard PIL format
        stride = surface.get_stride()
        with surface.get_data() as memory:
            pil_img = Image.frombuffer(
                "RGBA", (width, height), memory.tobytes(), "raw", "BGRA", stride
            )
            # Invoke system load copy to clear volatile pointer boundaries inside multi-process pipelines
            return pil_img.copy()

    @classmethod
    def download(cls, download_path: str, force: bool = False):
        download_p = Path(download_path)
        download_p.mkdir(parents=True, exist_ok=True)
        target_file = download_p / cls.FILENAME

        if not target_file.exists() or force:
            logger.info(
                f"Downloading MathWriting dataset from source: {cls.DOWNLOAD_URL}"
            )
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
                    desc="Downloading MathWriting",
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
        dataset_p = Path(dataset_path)
        archive_source = download_p / cls.FILENAME

        logger.info(f"Extracting package contents from {cls.FILENAME}...")
        if not (dataset_p / "readme.md").exists() or force:
            with tarfile.open(archive_source) as f:
                f.extractall(dataset_p)

            nested_folder = (
                dataset_p / "mathwriting-2024-excerpt"
                if cls.EXCERPT_MODE
                else dataset_p / "mathwriting-2024"
            )

            if nested_folder.exists() and nested_folder.is_dir():
                logger.info("Flattening nested package directory hierarchy...")
                for item in nested_folder.iterdir():
                    shutil.move(str(item), str(dataset_p / item.name))
                nested_folder.rmdir()
        else:
            raise FileExistsError(
                f"Extracted destination files already exist at {dataset_p / 'readme.md'}."
            )

        logger.info("MathWriting environment initialization finalized.")
