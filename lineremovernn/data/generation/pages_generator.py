"""
generate_pages.py — Synthetic ruled/clean page generator from proportional mixed datasets.
"""

from __future__ import annotations

import json
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image, ImageDraw
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
    """Instantiates completely independent dataset caches locally inside every spawned worker process."""
    global _WORKER_DATASETS, _WORKER_TOKENS
    _WORKER_TOKENS = mixed_tokens

    # Intentionally initialize inside the worker process context boundary to keep LRU/Dict caches OOM safe
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


def add_random_perspective(
    img: Image.Image, max_warp: float, rng: random.Random
) -> Image.Image:
    if max_warp <= 0:
        return img

    width, height = img.size
    src_points = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
    )
    max_dx, max_dy = width * max_warp, height * max_warp

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

    transformed = cv2.warpPerspective(
        np.array(img),
        final_matrix,
        (new_width, new_height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return Image.fromarray(transformed)


def _make_ink_word(img: Image.Image, target_h: int) -> Image.Image | None:
    orig_w, orig_h = img.size
    if orig_h == 0 or orig_w == 0:
        return None

    aspect_ratio = orig_w / orig_h

    # 1. Establish a realistic maximum height for complex vertical math.
    # A multi-tier fraction can occupy up to ~1.8x the height of a normal word
    # before it starts looking unnaturally large.
    if aspect_ratio < 1.2:  # Tall or square assets (fractions, integrals, matrices)
        max_allowable_h = int(target_h * 1.8)
    else:  # Short and wide assets (standard inline variables/equations)
        max_allowable_h = target_h

    # 2. Calculate the scaling factor based on this capped height target
    if orig_h > max_allowable_h:
        scale = max_allowable_h / orig_h
    else:
        scale = target_h / orig_h  # Don't upscale tiny assets unnecessarily

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
                step = max(1, W // 120)
                pts = [
                    (x, int(y_base + y_off + arc_y_offset(x, W, amplitude)))
                    for x in range(0, W, step)
                ]
                if len(pts) >= 2:
                    draw.line(pts, fill=darkness, width=lw)
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
    ruled_img.save(f"{ruled_dir}/{page_index}.jpg", quality=92)
    clean_img.save(f"{clean_dir}/{page_index}.jpg", quality=92)

    if labels_dir and labels:
        with open(f"{labels_dir}/{page_index}.json", "w", encoding="UTF-8") as f:
            json.dump(labels, f, indent=2)
    return page_index


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

    page_np = np.full((H, W), _PAPER_GRAY, dtype=np.uint8)

    # Unique entry offset for text compilation per page
    token_cursor = rng.randint(0, max(1, len(_WORKER_TOKENS) - 1))
    word_height = int(params.line_spacing * rng.uniform(0.58, 0.72))
    placed_words = []

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
            # Safely step through mixed token identifiers
            dset_id, asset_idx = _WORKER_TOKENS[token_cursor % len(_WORKER_TOKENS)]
            token_cursor += 1

            target_dataset = _WORKER_DATASETS[dset_id]

            # Grabs properties cleanly. Custom pre-processing happens internally inside the dataset subclass!
            try:
                word_img = target_dataset.get_image(asset_idx)
                asset_meta = target_dataset.assets[asset_idx]
            except Exception:
                continue

            if max_warp > 0.0:
                word_img = add_random_perspective(word_img, max_warp, rng)

            word_img = _make_ink_word(word_img, word_height)
            if word_img is None:
                continue

            ww, wh = word_img.size
            if x_cursor + ww > W - rng.randint(5, 30):
                break

            # 1. Calculate target coordinates on the main canvas
            py0, py1 = max(y_top, 0), min(y_top + wh, H)
            px0, px1 = max(x_cursor, 0), min(x_cursor + ww, W)

            # 2. Only proceed if there is a valid overlapping region
            if py1 > py0 and px1 > px0:
                word_np = np.array(word_img)

                # 3. Calculate exact corresponding slices relative to the local word crop
                wy0 = py0 - y_top
                wy1 = wy0 + (py1 - py0)
                wx0 = px0 - x_cursor
                wx1 = wx0 + (px1 - px0)

                # 4. Extract localized slices using the mapped variables
                mask = word_np[wy0:wy1, wx0:wx1, 3] > 0
                gray = word_np[wy0:wy1, wx0:wx1, :3].mean(axis=2).astype(np.uint8)

                # 5. Perfect 1:1 shape alignment guaranteed
                page_np[py0:py1, px0:px1][mask] = gray[mask]

                placed_words.append(
                    {
                        "text": asset_meta.text,
                        "x": int(px0),
                        "y": int(py0),
                        "w": int(px1 - px0),
                        "h": int(py1 - py0),
                    }
                )

            x_cursor += ww + params.word_gap + rng.randint(-2, 4)

    lines_layer = _draw_lines_layer(rng, W, H, params, use_arc, imperfect_lines)
    ruled_np = np.minimum(page_np, np.array(lines_layer))

    return (
        Image.fromarray(ruled_np).convert("RGB"),
        Image.fromarray(page_np).convert("RGB"),
        placed_words,
    )


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

    # Use defaults if no explicit configuration properties are supplied
    if not datasets:
        datasets = {1: IAMDataset}

    # Normalize properties and index counts via temporary local instances
    total_weight = sum(datasets.keys())
    normalized_sets = {
        f"ds_{i}": (cls, w / total_weight)
        for i, (w, cls) in enumerate(datasets.items())
    }

    logger.info("Assembling multi-dataset token layouts...")
    mixed_tokens: list[tuple[str, int]] = []
    dataset_blueprints: dict[str, type[ImageDataset]] = {}

    # Calculate target map pools (allocated at a large scale multiplier to handle distribution coverage)
    token_pool_target = max(100000, n * 40)

    for dset_key, (cls, weight) in normalized_sets.items():
        dataset_blueprints[dset_key] = cls

        # Instantiate a dry probe configuration simply to gather dataset length
        probe = cls(preload=False)
        dset_len = len(probe)
        if dset_len == 0:
            continue

        allocated_count = int(token_pool_target * weight)
        indices = random.choices(range(dset_len), k=allocated_count)
        mixed_tokens.extend([(dset_key, idx) for idx in indices])

    random.shuffle(mixed_tokens)
    logger.info(
        f"Assembled shared mixed tokens. Total token pool contains: {len(mixed_tokens)} slots."
    )

    with ProcessPoolExecutor(
        max_workers=cpu_workers,
        initializer=_worker_init,
        initargs=(dataset_blueprints, preload, mixed_tokens),
    ) as pool:
        futures = [
            pool.submit(
                _render_task,
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
        for fut in tqdm(
            as_completed(futures), total=n, desc="Generating mixed pages", unit="page"
        ):
            fut.result()

    logger.info("Done. Mixed generation pipeline completed successfully.")


if __name__ == "__main__":
    # Standard arg parser mapping to trigger standalone execution
    generate(n=20)
