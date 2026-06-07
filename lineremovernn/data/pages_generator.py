from __future__ import annotations

import json
import os
import random
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw
from PIL.Image import Resampling
from tqdm import tqdm

from lineremovernn.data.dataset import ImageDataset
from lineremovernn.data.iam import IAMDataset
from lineremovernn.data.pages import PagesDataset
from lineremovernn.utils import logging

logger = logging.get_logger("PageGenerator")

# ---------------------------------------------------------------------------
# Global Worker Context Layers
# ---------------------------------------------------------------------------

_WORKER_DATASETS: dict[str, ImageDataset] = {}
_WORKER_TOKENS: list[tuple[str, int]] = []


def _worker_init(
    dataset_blueprints: dict[str, type[ImageDataset]],
    preload: bool,
    mixed_tokens: list[tuple[str, int]],
) -> None:
    global _WORKER_DATASETS, _WORKER_TOKENS
    _WORKER_TOKENS = mixed_tokens
    _WORKER_DATASETS = {
        name: cls(preload=preload) for name, cls in dataset_blueprints.items()
    }


# ---------------------------------------------------------------------------
# Layout Parameter Models & Helpers
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
    w = 4000  # rng.randint(900, 2000)
    h = 4000  # rng.randint(1000, 2300)

    margin_left = int(w * rng.uniform(0.10, 0.18))
    margin_top = int(h * rng.uniform(0.06, 0.12))

    line_spacing = rng.randint(35, 100)
    word_gap = rng.randint(8, 25)

    n_lines = (h - margin_top) // line_spacing
    n_skipped = rng.randint(0, min(3, n_lines // 6))
    skipped = set(rng.sample(range(1, n_lines - 1), n_skipped)) if n_skipped else set()

    return PageParams(w, h, margin_left, margin_top, line_spacing, word_gap, skipped)


def add_random_perspective(
    img: Image.Image, max_warp: float, rng: random.Random
) -> Image.Image:
    if max_warp <= 0:
        return img

    width, height = img.size

    # 1. Map original image boundaries
    src_points = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
    )

    max_dx = width * max_warp
    max_dy = height * max_warp

    # 2. Compute random warp points
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

    # OPTIMIZATION 1: dst_points ARE the projected corners.
    # Extract bounding box limits instantly without matrix re-multiplication.
    min_x = dst_points[:, 0].min()
    max_x = dst_points[:, 0].max()
    min_y = dst_points[:, 1].min()
    max_y = dst_points[:, 1].max()

    new_width = max(1, int(max_x - min_x))
    new_height = max(1, int(max_y - min_y))

    # OPTIMIZATION 2: Apply translation in-place via algebraic row operations.
    # This completely avoids allocating a translation matrix and executing a matrix multiplication.
    matrix[0, :] -= min_x * matrix[2, :]
    matrix[1, :] -= min_y * matrix[2, :]

    # OPTIMIZATION 3: Use np.asarray() for a zero-copy memory view of the PIL state
    np_img = np.asarray(img)

    # OPTIMIZATION 4: Match border scalar to image layout to keep OpenCV out of slow multi-channel paths
    border_val = 0 if img.mode == "L" else (0, 0, 0, 0)

    transformed = cv2.warpPerspective(
        np_img,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,  # Switch to cv2.INTER_NEAREST for raw, maximum speed
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_val,
    )

    return Image.fromarray(transformed)


def _make_ink_word(img: Image.Image, target_h: int) -> Image.Image | None:
    orig_w, orig_h = img.size
    if orig_h == 0 or orig_w == 0:
        return None

    aspect_ratio = orig_w / orig_h
    max_allowable_h = int(target_h * 1.8) if aspect_ratio < 1.2 else target_h
    scale = max_allowable_h / orig_h if orig_h > max_allowable_h else target_h / orig_h

    new_w = max(1, int(orig_w * scale))
    new_h = max(1, int(orig_h * scale))

    return img.resize((new_w, new_h), Resampling.BILINEAR)


# ---------------------------------------------------------------------------
# Background Lines Drawing Execution
# ---------------------------------------------------------------------------


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

    # Pre-calculate vectorized x variables for arced lines once per page
    if use_arc:
        step = max(1, W // 120)
        x_vals = np.arange(0, W, step)
        t_vals = x_vals / W
        pi_t_vals = t_vals * np.pi
        pts_buffer = np.empty((len(x_vals) * 2,), dtype=np.int32)
        pts_buffer[0::2] = x_vals

    for x_v in range(0, W, params.line_spacing):
        draw.line(
            [(x_v, 0), (x_v, H)], fill=rng.randint(100, 180), width=rng.randint(1, 2)
        )

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
                # Vectorized calculations replace thousands of list comprehension loops
                y_offsets = amplitude * np.sin(pi_t_vals)
                y_vals = np.round(y_base + y_off + y_offsets).astype(np.int32)

                pts_buffer[1::2] = y_vals
                if len(pts_buffer) >= 4:
                    draw.line(pts_buffer.tolist(), fill=darkness, width=lw)
            else:
                draw.line(
                    [(0, int(y_base) + y_off), (W, int(y_base) + y_off)],
                    fill=darkness,
                    width=lw,
                )

    draw.line(
        [(params.margin_left, 0), (params.margin_left, H)],
        fill=rng.randint(140, 200),
        width=rng.randint(2, 3),
    )

    if imperfect_lines:
        for _ in range(rng.randint(40, 120)):
            hx, hy, hr = rng.randint(0, W), rng.randint(0, H), rng.randint(1, 4)
            draw.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], fill=255)

    return layer


# ---------------------------------------------------------------------------
# Core Rendering Task Thread Callables
# ---------------------------------------------------------------------------


def render_page(
    rng: random.Random,
    use_arc: bool = True,
    max_warp: float = 0.0,
    imperfect_lines: bool = False,
    page_index: int = 0,
) -> tuple[Image.Image, Image.Image, list[dict]]:
    params = random_page_params(rng)
    W, H = params.width, params.height
    n_lines = (H - params.margin_top) // params.line_spacing

    # Use native Pillow operations instead of numpy arrays
    page_img = Image.new("L", (W, H), _PAPER_GRAY)

    token_cursor = rng.randint(0, max(1, len(_WORKER_TOKENS) - 1))
    word_height = int(params.line_spacing * rng.uniform(0.58, 0.72))
    placed_words = []
    n_tokens = len(_WORKER_TOKENS)

    sum_fetching = 0
    sum_treating = 0
    sum_pasting = 0
    sum_json = 0
    x = 0

    for line_i in range(n_lines):
        if line_i in params.skipped:
            continue

        y_top = (
            (params.margin_top + line_i * params.line_spacing)
            - word_height
            - rng.randint(2, 6)
        )
        if y_top < 0:
            continue

        x_cursor = params.margin_left + rng.randint(0, 12)

        while x_cursor < W - 20:
            dset_id, asset_idx = _WORKER_TOKENS[token_cursor % n_tokens]
            token_cursor += 1

            t = time.time_ns()
            try:
                word_img = _WORKER_DATASETS[dset_id].get_image(asset_idx, mode="L")
                asset_meta = _WORKER_DATASETS[dset_id].assets[asset_idx]
            except Exception:
                logger.error(traceback.format_exc())
                continue
            sum_fetching += time.time_ns() - t
            t = time.time_ns()

            if max_warp > 0.0:
                word_img = add_random_perspective(word_img, max_warp, rng)

            word_img = _make_ink_word(word_img, word_height)
            if word_img is None:
                continue

            ww, wh = word_img.size
            if x_cursor + ww > W - rng.randint(5, 30):
                break

            py0, py1 = max(y_top, 0), min(y_top + wh, H)
            px0, px1 = max(x_cursor, 0), min(x_cursor + ww, W)

            if py1 > py0 and px1 > px0:
                # Calculate crops if word spills over page boundary
                wx0, wy0 = px0 - x_cursor, py0 - y_top
                wx1, wy1 = wx0 + (px1 - px0), wy0 + (py1 - py0)
                if wx0 == 0 and wy0 == 0 and wx1 == ww and wy1 == wh:
                    word_cropped = word_img
                else:
                    word_cropped = word_img.crop((wx0, wy0, wx1, wy1))

                sum_treating += time.time_ns() - t
                t = time.time_ns()

                # Highly optimized C-level pasting using the original image's alpha mask
                page_img.paste(word_cropped, (px0, py0), mask=word_cropped)
                sum_pasting += time.time_ns() - t
                t = time.time_ns()

                placed_words.append(
                    {
                        "text": asset_meta.text,
                        "x": int(px0),
                        "y": int(py0),
                        "w": int(px1 - px0),
                        "h": int(py1 - py0),
                    }
                )
                sum_json += time.time_ns() - t
                x += 1

            x_cursor += ww + params.word_gap + rng.randint(-2, 4)
    t = time.time_ns()

    lines_layer = _draw_lines_layer(rng, W, H, params, use_arc, imperfect_lines)
    time_draw_lines = time.time_ns() - t

    # C-optimized combination instead of numpy.minimum
    t = time.time_ns()
    ruled_img = ImageChops.darker(page_img, lines_layer).convert("L")
    clean_img = page_img.convert("L")
    clean_t = time.time_ns() - t
    logger.debug(
        f"Avg fetching {sum_fetching / x / 1_000_000}ms, avg treating {sum_treating / x / 1_000_000}ms,"
        + f" avg pasting {sum_pasting / x / 1_000_000}ms, avg json {sum_json / x / 1_000_000}ms, draw lines {time_draw_lines / 1_000_000}, clean {clean_t / 1_000_000}"
    )

    return ruled_img, clean_img, placed_words


def _render_task_wrapper(args):
    """Unpacks arguments to allow process map chunking."""
    return _render_task(*args)


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
        rng,
        use_arc=use_arc,
        max_warp=max_warp,
        imperfect_lines=imperfect_lines,
        page_index=page_index,
    )
    logger.debug("Saving")
    ruled_img.save(f"{ruled_dir}/{page_index}.jpg", quality=92)
    clean_img.save(f"{clean_dir}/{page_index}.jpg", quality=92)

    if labels_dir and labels:
        with open(f"{labels_dir}/{page_index}.json", "w", encoding="UTF-8") as f:
            json.dump(labels, f, indent=2)
    return page_index


# ---------------------------------------------------------------------------
# Standard Orchestration Routine
# ---------------------------------------------------------------------------


def generate(
    n: int = 50,
    preload: bool = False,
    use_arc: bool = True,
    max_warp: float = 0.0,
    imperfect_lines: bool = False,
    save_json: bool = False,
    seed: int | None = None,
    target: Path | None = None,
    workers: int | None = None,
    datasets: dict[float, type[ImageDataset]] | None = None,
) -> None:
    cpu_workers = workers or os.cpu_count() or 4
    target = target or PagesDataset.path()

    ruled_dir, clean_dir = target / "ruled-pages", target / "clean-pages"
    ruled_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = target / "labels" if save_json else None
    if labels_dir:
        labels_dir.mkdir(parents=True, exist_ok=True)

    if not datasets:
        datasets = {1: IAMDataset}

    total_weight = sum(datasets.keys())
    normalized_sets = {
        f"ds_{i}": (cls, w / total_weight)
        for i, (w, cls) in enumerate(datasets.items())
    }

    logger.info("Assembling multi-dataset token layouts...")
    mixed_tokens: list[tuple[str, int]] = []
    dataset_blueprints: dict[str, type[ImageDataset]] = {}
    token_pool_target = max(100000, n * 40)

    for dset_key, (cls, weight) in normalized_sets.items():
        dataset_blueprints[dset_key] = cls
        probe = cls(preload=False)
        dset_len = len(probe)
        if dset_len == 0:
            continue

        allocated_count = int(token_pool_target * weight)
        indices = random.choices(range(dset_len), k=allocated_count)
        mixed_tokens.extend([(dset_key, idx) for idx in indices])

    random.shuffle(mixed_tokens)
    logger.info(
        f"Assembled shared mixed tokens. Pool contains: {len(mixed_tokens)} slots."
    )

    # Calculate optimal chunk size to lower process IPC overhead
    chunk_size = max(1, n // (cpu_workers * 4))

    tasks = [
        (
            i,
            seed,
            use_arc,
            max_warp,
            imperfect_lines,
            str(ruled_dir),
            str(clean_dir),
            str(labels_dir) if labels_dir else None,
        )
        for i in range(n)
    ]

    with ProcessPoolExecutor(
        max_workers=cpu_workers,
        initializer=_worker_init,
        initargs=(dataset_blueprints, preload, mixed_tokens),
    ) as pool:
        # Use map to push large batches down the pipeline concurrently
        for _ in tqdm(
            pool.map(_render_task_wrapper, tasks, chunksize=chunk_size),
            total=n,
            desc="Generating mixed pages",
            unit="page",
        ):
            pass

    logger.info("Done. Mixed generation pipeline completed successfully.")


if __name__ == "__main__":
    generate(n=20)
