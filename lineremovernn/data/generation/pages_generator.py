"""
generate_pages.py — Synthetic ruled/clean page generator from IAM word crops.

Usage:
    python generate_pages.py [--n 50] [--preload] [--arc] [--max-warp 0.05] [--imperfect-lines] [--save-json] [--seed 42] [--workers 4]

Outputs:
    <target>/ruled-pages/0.jpg, 1.jpg, ..., n.jpg
    <target>/clean-pages/0.jpg,  1.jpg, ..., n.jpg
    <target>/labels/0.json, 1.json, ..., n.json (Optional)
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image, ImageDraw
from PIL.Image import Resampling
from tqdm import tqdm

from lineremovernn.data.iam import IAMDataset
from lineremovernn.data.pages import PagesDataset
from lineremovernn.utils import logging

logger = logging.get_logger("PageGenerator")


# ---------------------------------------------------------------------------
# Data loading & Dual-Caching Strategy
# ---------------------------------------------------------------------------

WordEntry = tuple[str, tuple[int, int, int, int], str, int]


def load_words(iam_path: Path) -> list[WordEntry]:
    words: list[WordEntry] = []
    words_file = iam_path / "words.txt"
    with open(words_file, encoding="UTF-8") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("#") or len(line.split(" ")) != 9:
                continue
            filename, segmentation, gray, x, y, w, h, typ, transcript = line.split(" ")
            if segmentation == "err":
                continue
            parts = filename.split("-")
            path = (
                iam_path / "words" / parts[0] / "-".join(parts[:2]) / f"{filename}.png"
            )
            words.append(
                (str(path), (int(x), int(y), int(w), int(h)), transcript, int(gray))
            )
    return words


def _read_one(path: str) -> tuple[str, bytes | None]:
    """Used strictly during global preloading step."""
    try:
        with open(path, "rb") as f:
            return path, f.read()
    except Exception:
        return path, None


@lru_cache(maxsize=3000)
def _worker_load_image_bytes(path: str) -> bytes | None:
    """Bounded cache used within individual workers when upfront preloading is disabled."""
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return None


def preload_images(words: list[WordEntry], io_workers: int = 16) -> dict[str, bytes]:
    """Preloads the entire dataset into a single dictionary using multi-threading."""
    paths = list({w[0] for w in words})
    cache: dict[str, bytes] = {}

    with ThreadPoolExecutor(max_workers=io_workers) as pool:
        futures = {pool.submit(_read_one, p): p for p in paths}
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Preloading images",
            unit="img",
        ):
            p, data = fut.result()
            if data is not None:
                cache[p] = data

    logger.info(
        "Preloaded %d images (%.1f MB)",
        len(cache),
        sum(len(v) for v in cache.values()) / 1e6,
    )
    return cache


# ---------------------------------------------------------------------------
# Page layout parameters
# ---------------------------------------------------------------------------


class PageParams(NamedTuple):
    width: int
    height: int
    margin_left: int
    margin_top: int
    line_spacing: int
    word_gap: int
    skipped: set[int]


_PAPER_GRAY = 250


def random_page_params(rng: random.Random) -> PageParams:
    w = rng.randint(900, 2000)
    h = rng.randint(1000, 2300)

    margin_left = int(w * rng.uniform(0.10, 0.18))
    margin_top = int(h * rng.uniform(0.06, 0.12))

    line_spacing = rng.randint(35, 100)
    word_gap = rng.randint(8, 25)

    n_lines = (h - margin_top) // line_spacing
    n_skipped = rng.randint(0, min(3, n_lines // 6))
    skipped = set(rng.sample(range(1, n_lines - 1), n_skipped)) if n_skipped else set()

    return PageParams(w, h, margin_left, margin_top, line_spacing, word_gap, skipped)


def arc_y_offset(x: int, page_width: int, amplitude: float) -> float:
    t = x / page_width
    return amplitude * math.sin(t * math.pi)


# ---------------------------------------------------------------------------
# Image Augmentations
# ---------------------------------------------------------------------------


def add_random_perspective(
    img: Image.Image, max_warp: float, rng: random.Random
) -> Image.Image:
    if max_warp <= 0:
        return img

    width, height = img.size
    src_points = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
    )

    max_dx = width * max_warp
    max_dy = height * max_warp

    dst_points = np.array(
        [
            [rng.uniform(-max_dx, max_dx), rng.uniform(-max_dy, max_dy)],
            [width + rng.uniform(-max_dx, max_dx), rng.uniform(-max_dy, max_dy)],
            [
                width + rng.uniform(-max_dx, max_dx),
                height + rng.uniform(-max_dy, max_dy),
            ],
            [rng.uniform(-max_dx, max_dx), height + rng.uniform(-max_dy, max_dy)],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    corners = np.array(
        [[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]], dtype=np.float32
    ).T
    new_corners = matrix @ corners
    new_corners /= new_corners[2]

    min_x, min_y = new_corners[:2].min(axis=1)
    max_x, max_y = new_corners[:2].max(axis=1)
    new_width, new_height = max(1, int(max_x - min_x)), max(1, int(max_y - min_y))

    translation = np.array(
        [[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float32
    )
    final_matrix = translation @ matrix

    img_np = np.array(img)
    transformed = cv2.warpPerspective(
        img_np,
        final_matrix,
        (new_width, new_height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    return Image.fromarray(transformed)


# ---------------------------------------------------------------------------
# Process-pool worker initializer
# ---------------------------------------------------------------------------

_worker_cache: dict[str, bytes] | None = None
_worker_words: list[WordEntry] = []


def _worker_init(words: list[WordEntry], cache: dict[str, bytes] | None) -> None:
    global _worker_cache, _worker_words
    _worker_words = words
    _worker_cache = cache


def _render_task(
    page_index: int,
    seed: int | None,
    use_arc: bool,
    max_warp: float,
    imperfect_lines: bool,
    ruled_dir: str,
    clean_dir: str,
    labels_dir: str | None,
) -> int:
    rng = random.Random(None if seed is None else seed + page_index)
    ruled_img, clean_img, labels = render_page(
        _worker_words,
        _worker_cache,
        rng,
        use_arc=use_arc,
        max_warp=max_warp,
        imperfect_lines=imperfect_lines,
    )
    ruled_img.save(f"{ruled_dir}/{page_index}.jpg", quality=92)
    clean_img.save(f"{clean_dir}/{page_index}.jpg", quality=92)

    if labels_dir and labels:
        with open(f"{labels_dir}/{page_index}.json", "w", encoding="UTF-8") as f:
            json.dump(labels, f, indent=2)

    return page_index


# ---------------------------------------------------------------------------
# Core page renderer
# ---------------------------------------------------------------------------


def _open_word_image(raw: bytes) -> Image.Image | None:
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGBA")

        orig_w, orig_h = img.size
        if orig_w > 4 and orig_h > 4:
            img = img.crop((2, 2, orig_w - 2, orig_h - 2))

        data = np.array(img)
        brightness = data[..., :3].mean(axis=2)

        bg_mask = brightness > 160

        data[bg_mask] = [255, 255, 255, 0]
        data[~bg_mask, 3] = 255

        return Image.fromarray(data, "RGBA")
    except Exception:
        return None


def _make_ink_word(img: Image.Image, target_h: int) -> Image.Image | None:
    orig_w, orig_h = img.size
    if orig_h == 0:
        return None
    scale = target_h / orig_h
    new_w = max(1, int(orig_w * scale))
    new_h = target_h
    return img.resize((new_w, new_h), Resampling.BILINEAR)


def _draw_lines_layer(
    rng: random.Random,
    W: int,
    H: int,
    params: PageParams,
    use_arc: bool,
    imperfect_lines: bool,
) -> Image.Image:
    layer = Image.new("L", (W, H), 255)
    draw = ImageDraw.Draw(layer)

    n_lines = (H - params.margin_top) // params.line_spacing
    sub = rng.randint(3, 6)

    for x_v in range(0, W, params.line_spacing):
        v_darkness = rng.randint(100, 180)
        draw.line([(x_v, 0), (x_v, H)], fill=v_darkness, width=rng.randint(1, 2))

    for i in range(n_lines + 1):
        y_group = params.margin_top + i * params.line_spacing
        sub_step = params.line_spacing / sub

        for j in range(sub):
            y_base = y_group + j * sub_step
            is_main = j in (0, sub - 1)
            darkness = rng.randint(100, 180) if is_main else rng.randint(200, 240)
            lw = rng.randint(1, 3) if is_main else 1
            y_off = rng.randint(-3, 3)

            if use_arc:
                amplitude = rng.uniform(-15.0, 15.0)
                step = max(1, W // 120)
                pts = []
                for x in range(0, W, step):
                    y = y_base + y_off + arc_y_offset(x, W, amplitude)
                    pts.append((x, int(y)))
                if len(pts) >= 2:
                    draw.line(pts, fill=darkness, width=lw)
            else:
                y_int = int(y_base) + y_off
                draw.line([(0, y_int), (W, y_int)], fill=darkness, width=lw)

    draw.line(
        [(params.margin_left, 0), (params.margin_left, H)],
        fill=rng.randint(140, 200),
        width=rng.randint(2, 3),
    )

    if imperfect_lines:
        for _ in range(rng.randint(40, 120)):
            hx = rng.randint(0, W)
            hy = rng.randint(0, H)
            hr = rng.randint(1, 4)
            draw.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], fill=255)

    return layer


def render_page(
    words: list[WordEntry],
    cache: dict[str, bytes] | None,
    rng: random.Random,
    use_arc: bool = True,
    max_warp: float = 0.0,
    imperfect_lines: bool = False,
) -> tuple[Image.Image, Image.Image, list[dict]]:
    params = random_page_params(rng)
    W, H = params.width, params.height
    n_lines = (H - params.margin_top) // params.line_spacing

    page_np = np.full((H, W), _PAPER_GRAY, dtype=np.uint8)

    start_offset = rng.randint(0, max(0, len(words) - 1)) if words else 0
    word_idx = 0
    word_height = int(params.line_spacing * rng.uniform(0.58, 0.72))

    placed_words = []

    for line_i in range(n_lines):
        if line_i in params.skipped:
            continue

        y_base = params.margin_top + line_i * params.line_spacing
        y_top = y_base - word_height - rng.randint(2, 6)
        if y_top < 0:
            continue

        x_cursor = params.margin_left + rng.randint(0, 12)

        while x_cursor < W - 20 and word_idx < len(words):
            entry = words[(start_offset + word_idx) % len(words)]
            word_idx += 1

            # Determine whether to read from global dictionary cache or local on-demand cache
            if cache is not None:
                raw = cache.get(entry[0])
            else:
                raw = _worker_load_image_bytes(entry[0])

            if raw is None:
                continue

            word_img = _open_word_image(raw)
            if word_img is None:
                continue

            if max_warp > 0.0:
                word_img = add_random_perspective(word_img, max_warp, rng)

            word_img = _make_ink_word(word_img, word_height)
            if word_img is None:
                continue

            ww, wh = word_img.size
            if x_cursor + ww > W - rng.randint(5, 30):
                break

            y_jitter = rng.randint(-3, 3)
            paste_y = y_top + y_jitter

            word_np = np.array(word_img)
            alpha = word_np[..., 3]
            gray = word_np[..., :3].mean(axis=2).astype(np.uint8)

            py0 = max(paste_y, 0)
            py1 = min(paste_y + wh, H)
            px0 = max(x_cursor, 0)
            px1 = min(x_cursor + ww, W)
            wy0 = py0 - paste_y
            wy1 = wy0 + (py1 - py0)
            wx0 = px0 - x_cursor
            wx1 = wx0 + (px1 - px0)

            if py1 > py0 and px1 > px0:
                mask = alpha[wy0:wy1, wx0:wx1] > 0
                page_np[py0:py1, px0:px1][mask] = gray[wy0:wy1, wx0:wx1][mask]

                placed_words.append(
                    {
                        "text": entry[2],
                        "x": int(px0),
                        "y": int(py0),
                        "w": int(px1 - px0),
                        "h": int(py1 - py0),
                    }
                )

            x_cursor += ww + params.word_gap + rng.randint(-2, 4)

    lines_layer = _draw_lines_layer(rng, W, H, params, use_arc, imperfect_lines)
    lines_np = np.array(lines_layer)

    ruled_np = np.minimum(page_np, lines_np)

    clean_img = Image.fromarray(page_np).convert("RGB")
    ruled_img = Image.fromarray(ruled_np).convert("RGB")

    return ruled_img, clean_img, placed_words


# ---------------------------------------------------------------------------
# Main orchestrating entry point
# ---------------------------------------------------------------------------


def generate(
    n: int = 50,
    preload: bool = False,
    use_arc: bool = True,
    max_warp: float = 0.0,
    imperfect_lines: bool = False,
    save_json: bool = False,
    seed: int | None = None,
    iam_path: Path | None = None,
    target: Path | None = None,
    workers: int | None = None,
    io_workers: int = 16,
) -> None:
    cpu_workers = workers or os.cpu_count() or 4

    iam_path = iam_path or IAMDataset.path()
    target = target or PagesDataset.path()

    ruled_dir = target / "ruled-pages"
    clean_dir = target / "clean-pages"
    ruled_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    labels_dir = target / "labels" if save_json else None
    if labels_dir:
        labels_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading IAM dataset structural indexes from %s", iam_path)
    words = load_words(iam_path)
    if not words:
        raise RuntimeError("No words loaded — check iam_path.")

    # Filter out broken or missing paths cleanly without allocating byte caches
    logger.info("Validating dataset paths on file system...")
    words = [w for w in words if os.path.exists(w[0])]

    cache: dict[str, bytes] | None = None
    if preload:
        logger.info("Upfront global preloading enabled. Warning: High RAM Usage.")
        cache = preload_images(words, io_workers=io_workers)
        if not cache:
            raise RuntimeError("Image cache built empty — check word paths.")
    else:
        logger.info("Using memory-bounded worker-local LRU caches (OOM Safe Mode).")

    logger.info(
        "Generating %d page pairs with %d CPU workers → %s",
        n,
        cpu_workers,
        target,
    )

    ruled_str = str(ruled_dir)
    clean_str = str(clean_dir)
    labels_str = str(labels_dir) if labels_dir else None

    with ProcessPoolExecutor(
        max_workers=cpu_workers,
        initializer=_worker_init,
        initargs=(words, cache),
    ) as pool:
        futures = [
            pool.submit(
                _render_task,
                i,
                seed,
                use_arc,
                max_warp,
                imperfect_lines,
                ruled_str,
                clean_str,
                labels_str,
            )
            for i in range(n)
        ]
        for fut in tqdm(
            as_completed(futures),
            total=n,
            desc="Generating pages",
            unit="page",
        ):
            fut.result()

    logger.info("Done. Execution completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic ruled/clean page pairs with optional warping and labeling features."
    )
    parser.add_argument(
        "--n", type=int, default=50, help="Number of page pairs to generate"
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Preload entire image collection into RAM upfront (risks OOM crashes on small configurations)",
    )
    parser.add_argument(
        "--arc", action="store_true", help="Use slightly arced ruled lines"
    )
    parser.add_argument(
        "--max-warp",
        type=float,
        default=0.05,
        help="Maximum perspective warp factor for word crops (0.0 to disable)",
    )
    parser.add_argument(
        "--imperfect-lines",
        action="store_true",
        help="Inject tiny structural imperfections and gaps into rules",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Export ground-truth word layout coordinates as JSON files",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="RNG seed for reproducibility"
    )
    parser.add_argument(
        "--iam", type=Path, default=None, help="Override IAM dataset path"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Override output target path"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="CPU worker processes (default: all cores)",
    )
    parser.add_argument(
        "--io-workers",
        type=int,
        default=16,
        help="I/O threads for upfront preloading step (default: 16)",
    )
    args = parser.parse_args()

    generate(
        n=args.n,
        preload=args.preload,
        use_arc=args.arc,
        max_warp=args.max_warp,
        imperfect_lines=args.imperfect_lines,
        save_json=args.save_json,
        seed=args.seed,
        iam_path=args.iam,
        target=args.out,
        workers=args.workers,
        io_workers=args.io_workers,
    )
