import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import tqdm
from PIL import Image

from lineremovernn.data.dataset import CropAsset, DownloadableDataset, ImageDataset
from lineremovernn.utils import logging

logger = logging.get_logger("IAM")

WordEntry = tuple[str, tuple[int, int, int, int], str, int]


class IAMDataset(DownloadableDataset, ImageDataset):
    ID = "IAM"

    def __init__(self, preload: bool = False):
        # Initialize parent classes explicitly if cooperative MRO is unstable
        DownloadableDataset.__init__(self)
        ImageDataset.__init__(self, preload=preload)

        # Automatically pull data into scope upon instantiation if files exist
        if self.available():
            self._load_metadata()

    def _load_metadata(self) -> None:
        """Parses words mapping data into structural CropAsset arrays."""
        words_file = self.path() / "words.txt"
        if not words_file.exists():
            logger.warning(
                f"Metadata index not found at {words_file}. Run install() first."
            )
            return

        with open(words_file, encoding="UTF-8") as f:
            for line in f:
                line = line.rstrip()
                if line.startswith("#") or len(line.split(" ")) != 9:
                    continue
                filename, segmentation, _, _, _, _, _, _, transcript = line.split(" ")
                if segmentation == "err":
                    continue

                parts = filename.split("-")
                img_path = (
                    self.path()
                    / "words"
                    / parts[0]
                    / "-".join(parts[:2])
                    / f"{filename}.png"
                )

                # Exclusively utilize the standardized self.assets stream
                self.assets.append(CropAsset(path=str(img_path), text=transcript))

        if self.preload_enabled:
            self.preload()

    # @lru_cache(maxsize=200)
    def get_image(self, idx: int, mode="RGBA") -> Image.Image:
        img = ImageDataset.get_image(self, idx)

        # Crop logic
        orig_w, orig_h = img.size
        if orig_w > 4 and orig_h > 4:
            img = img.crop((2, 2, orig_w - 2, orig_h - 2))

        # 1. Always convert to L (grayscale) first for fast processing
        gray = np.array(img.convert("L"))
        h, w = gray.shape

        # 2. Apply background masking on the 2D array
        # This is the fastest way to handle the logic
        bg_mask = gray > 160
        gray[bg_mask] = 255  # Set background to white

        # 3. Handle mode output
        if mode == "L":
            return Image.fromarray(gray, mode="L")

        elif mode == "RGBA":
            # Expand to 4 channels only if requested
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            # Fill RGB channels with gray value
            rgba[..., 0:3] = gray[..., np.newaxis]
            # Set Alpha channel: Ink=255, Background=0
            rgba[..., 3] = np.where(bg_mask, 0, 255)

            return Image.fromarray(rgba, mode="RGBA")

        else:
            # Fallback for unexpected modes
            return Image.fromarray(gray, mode="L").convert(mode)

    @classmethod
    def download(cls, download_path: str, force: bool = False):
        dataset_path = Path(download_path) / "IAM_Words"
        dataset_path.mkdir(parents=True, exist_ok=True)

        source_url = "https://github.com/sayakpaul/Handwriting-Recognizer-in-Keras/releases/download/v1.0.0/IAM_Words.zip"

        archive_target = dataset_path / "words.tgz"
        if not archive_target.exists() or force:
            logger.info("Downloading IAM dataset repository...")
            cls._download_and_unzip(source_url, dataset_path)
        else:
            raise FileExistsError(
                f"Dataset already exists at {dataset_path}. Use force=True to overwrite."
            )

    @classmethod
    def extract(cls, download_path: str, dataset_path: str, force: bool = False):
        target_root = Path(dataset_path)
        dl_source_dir = Path(download_path) / "IAM_Words" / "IAM_Words"

        words_tgz = dl_source_dir / "words.tgz"
        words_txt = dl_source_dir / "words.txt"

        logger.info("Extracting word snippets from words.tgz...")
        words_out_dir = target_root / "words"
        if not words_out_dir.exists() or force:
            with tarfile.open(words_tgz) as f:
                f.extractall(words_out_dir)
        else:
            raise FileExistsError(
                f"Extracted directory already exists at {words_out_dir}."
            )

        logger.info("Staging words.txt context into root location...")
        txt_out_target = target_root / "words.txt"
        if not txt_out_target.exists() or force:
            words_txt.rename(txt_out_target)
        else:
            raise FileExistsError(f"Metadata file already exists at {txt_out_target}.")

        logger.info("Dataset staging completed successfully.")

    @classmethod
    def _download_and_unzip(
        cls, url: str, extract_to: Path, chunk_size: int = 1024 * 1024
    ) -> None:
        extract_to.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=True) as tmp_file:
            logger.info("Establishing secure downstream connection...")
            with urlopen(url) as response:
                total_size = int(response.headers.get("Content-Length", 0))

                with tqdm.tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc="Downloading Stream",
                    leave=True,
                ) as pbar:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        tmp_file.write(chunk)
                        pbar.update(len(chunk))

            tmp_file.flush()
            tmp_file.seek(0)

            logger.info(f"Unpacking file manifest into {extract_to}...")
            with ZipFile(tmp_file) as zf:
                zf.extractall(path=extract_to)
