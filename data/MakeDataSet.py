import shutil
import PIL.Image as Image
import PIL.ImageDraw
import PIL.ImageOps
import cv2
import numpy as np
from random import randint, random, uniform as randfloat
import tqdm
from functools import lru_cache
from multiprocessing import Pool, Value
from IAM import split_into_blocks
from torchvision.transforms import ToTensor, ToPILImage
import math
import time
import json
import os
import argparse


# Load word metadata
words = []
with open("./words.txt", encoding="UTF-8", mode="r") as words_file:
    for line in words_file:
        line = line.rstrip()
        if line.startswith("#") or len(line.split(" ")) > 9:
            continue
        filename, segmentation, grayScale, x, y, w, h, typ, transcript = line.split(" ")
        if segmentation == "err":
            continue
        fnamesplit = filename.split("-")
        path = f"./words/{fnamesplit[0]}/{'-'.join(fnamesplit[:2])}/{filename}.png"
        words.append(
            (path, (int(x), int(y), int(w), int(h)), transcript, int(grayScale))
        )

data_dir = "./"


# Transparent background function
def makeTransparentBG(img):
    """Make the white background of an image transparent."""
    img_data = np.array(img)
    avg = img_data[..., :3].mean(axis=2)
    mask = avg <= 200
    img_data[..., 3] = mask * 255
    return PIL.Image.fromarray(img_data, "RGBA")


@lru_cache(maxsize=1000)
def load_image(path) -> Image.Image:
    """Load and preprocess a word image."""
    try:
        img = PIL.Image.open(path).convert("RGBA")
        return makeTransparentBG(img)
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None


def draw_imperfect_arc(
    lines_draw,
    draw,
    bbox,
    start,
    end,
    fill=(0, 0, 0),
    width=2,
    hole_count=100,
    hole_size_range=(2, 14),
    outgrowth_count=100,
    outgrowth_size_range=(1, 10),
    mask=None,
):
    """
    Draws an imperfect arc with holes and outgrowths, and creates a mask to avoid overlap with text.

    :param draw: Pillow ImageDraw object
    :param bbox: Bounding box for the arc
    :param start: Start angle of the arc
    :param end: End angle of the arc
    :param fill: Fill color of the arc
    :param width: Width of the arc
    :param hole_count: Number of holes to create
    :param outgrowth_count: Number of outgrowths to add
    :param mask: Mask to apply the holes (to avoid interfering with words)
    """
    # Draw the base arc
    lines_draw.arc(bbox, start=start, end=end, fill=fill, width=width)
    uniform = randfloat
    # If a mask is provided, we will add holes to it
    for _ in range(hole_count):
        angle = math.radians(uniform(start, end))
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        radius_x = (bbox[2] - bbox[0]) / 2
        radius_y = (bbox[3] - bbox[1]) / 2

        hole_x = center_x + radius_x * math.cos(angle)
        hole_y = center_y + radius_y * math.sin(angle)
        hole_size = randint(*hole_size_range)  # Random hole size
        draw.ellipse(
            [
                hole_x - hole_size,
                hole_y - hole_size,
                hole_x + hole_size,
                hole_y + hole_size,
            ],
            fill="white",
        )

    # Add outgrowths to the main image
    for _ in range(outgrowth_count):
        angle = math.radians(uniform(start, end))
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        radius_x = (bbox[2] - bbox[0]) / 2
        radius_y = (bbox[3] - bbox[1]) / 2

        outgrowth_x = center_x + radius_x * math.cos(angle)
        outgrowth_y = center_y + radius_y * math.sin(angle)
        outgrowth_size = randint(*outgrowth_size_range)  # Random outgrowth size
        draw.line(
            [
                outgrowth_x,
                outgrowth_y,
                outgrowth_x + uniform(-outgrowth_size, outgrowth_size),
                outgrowth_y + uniform(-outgrowth_size, outgrowth_size),
            ],
            fill=fill,
            width=1,
        )


def draw_arc_with_condition(
    has_imperfect_arcs, draw, lines_draw, mask_draw, bbox, start, end, fill, width
):
    if has_imperfect_arcs:
        return draw_imperfect_arc(lines_draw, mask_draw, bbox, start, end, fill, width)
    else:
        return draw.arc(bbox, start=start, end=end, fill=fill, width=width)


def add_random_perspective(img: Image.Image, max_warp=0.3):
    """
    Apply a random perspective transformation to an image without cropping.

    :param img: PIL Image
    :param max_warp: Maximum proportion of width/height to perturb the corners
    :return: Transformed PIL Image
    """
    width, height = img.size

    # Define the original corner points
    src_points = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
    )

    # Generate random perturbations within a safe range
    def random_offset(val, max_offset):
        return val + randfloat(-max_offset, max_offset)

    max_dx, max_dy = width * max_warp, height * max_warp

    dst_points = np.array(
        [
            [random_offset(0, max_dx), random_offset(0, max_dy)],
            [random_offset(width, max_dx), random_offset(0, max_dy)],
            [random_offset(width, max_dx), random_offset(height, max_dy)],
            [random_offset(0, max_dx), random_offset(height, max_dy)],
        ],
        dtype=np.float32,
    )

    # Compute the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Compute the new bounding box
    corners = np.array(
        [[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]], dtype=np.float32
    ).T

    new_corners = matrix @ corners
    new_corners /= new_corners[2]  # Normalize

    # Compute new image size
    min_x, min_y = new_corners[:2].min(axis=1)
    max_x, max_y = new_corners[:2].max(axis=1)

    new_width, new_height = int(max_x - min_x), int(max_y - min_y)

    # Adjust transformation to keep everything visible
    translation = np.array(
        [[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float32
    )

    final_matrix = translation @ matrix

    # Apply transformation
    img_np = np.array(img)
    transformed = cv2.warpPerspective(
        img_np, final_matrix, (new_width, new_height), borderMode=cv2.BORDER_CONSTANT
    )

    return Image.fromarray(transformed)


def make_page(args):
    global blocks_index, blocks_index_l
    """Generate a single page with random words."""
    imageIndex, split, extended, dirs, size = args
    pages_dir, pages_dir_blocks, nolines_dir, nolines_dir_blocks, json_dir = dirs
    # randomPageSize = lambda: (
    #    size[0] if size[0] != 0 else (3000 if not extended else randint(1000, 4000))
    # )
    pageWidth, pageHeight = 2500, 3500  # randomPageSize(), randomPageSize()
    # Create a blank page with white background
    page = PIL.Image.new(
        mode="RGBA", size=(pageWidth, pageHeight), color=(255, 255, 255)
    )
    draw = PIL.ImageDraw.Draw(page)
    mask = PIL.Image.new("L", (pageWidth, pageHeight), 0)  # Create a mask for holes
    mask_draw = PIL.ImageDraw.Draw(mask)
    linesImg = PIL.Image.new(
        "L", (pageWidth, pageHeight), 0
    )  # Create a blank image for lines
    lines_draw = PIL.ImageDraw.Draw(linesImg)

    # Initialize variables for positioning
    randX = lambda: (
        30
        if not extended
        else (randint(30, 50) if bool(randint(0, 1)) else randint(100, 120))
    )
    x = randX()
    y = randint(0, 50)
    lineSize = randint(120, 200)  # Random line height for text
    smallLinesByLine = randint(4, 7)
    smallLineSize = (
        lineSize / smallLinesByLine
    )  # Distance between small horizontal lines
    lines = []  # List to store y-coordinates of lines
    wordsL = []  # List to store metadata about the words on the page
    gS = 255  # Grayscale value (used for line color)
    hasImperfectArcs = False  # bool(randint(0, 1)) if extended else False  # Randomly decide whether to add arcs
    skipLineProb = (
        1 / 4 if extended else 0
    )  # Probability of skipping one or multiple lines
    stopX = pageWidth - randX()

    # Generate content by iterating through the words list
    i = randint(0, len(words))
    max_h = 0
    while True:
        word = words[i % len(words)]  # Cycle through the words list
        path, box, transcript, wgS = word  # Unpack word details

        # Update grayscale to the darkest word color so far
        if wgS < gS:
            gS = wgS

        # Extract dimensions of the word box
        _, _, w, h = box

        max_h = max(max_h, h)

        # Load the word image
        wordimg = load_image(path)
        if wordimg is None or wordimg.width == 0 or wordimg.height == 0:
            words.pop(i % len(words))
            continue
        randResize = lambda x: int(randfloat(0.7, 1.3) * x)
        w = randResize(wordimg.width)
        h = randResize(wordimg.height)
        if not w > 0 or not h > 0:
            continue
        wordimg = wordimg.resize((w, h))
        wordimg = add_random_perspective(wordimg)
        i += 1

        # Move to a new line if the word doesn't fit in the current row
        if x + w >= stopX:
            x = randX()
            stopX = pageWidth - randX()
            lines.append(y)  # Add current y-coordinate to lines list
            y += (
                max(max_h, lineSize)
                if randint(0, 1) < skipLineProb
                else max(max_h, lineSize) * randint(1, 2)
            )
            max_h = 0

        # Stop adding words if the page height is exceeded
        if y >= pageHeight - lineSize:
            break

        # Random vertical adjustment for the word placement
        rn = randint(-20, 20)

        # Add metadata about the word to wordsL list
        wordsL.append({"text": transcript, "x": x, "y": y - h + rn, "w": w, "h": h})

        # Paste the word image onto the page
        # wordimg = apply_perspective_transform(wordimg)
        page.paste(wordimg, (x, y - h + rn), wordimg)

        # Update x-coordinate for the next word, with a random gap
        x += w + randint(-10, 30)

    # Save the page without lines to a separate directory
    page.save(f"{nolines_dir}/{imageIndex}-page.png")
    if split:
        blocks = split_into_blocks(ToTensor()(page))
        for i, block in enumerate(blocks):
            with blocks_index.get_lock():
                blocks_index.value += 1
                block_number = blocks_index.value - 1
            ToPILImage()(block).convert("L").save(
                os.path.join(nolines_dir_blocks, f"{block_number}.png"),
                format="PNG",
            )

    # Save word metadata to a JSON file
    with open(f"{json_dir}/{imageIndex}.json", mode="w") as jsonfile:
        json.dump(wordsL, jsonfile)

        # Draw horizontal arc-like lines at the stored y-coordinates
    TogglingChance = 5  # So 1/TogglingChance of changing direction of arcs
    arcToggle = False
    for y in lines:
        for j in range(smallLinesByLine):
            offset = randint(-5, 5)  # Slight randomness for arcs
            arc_height = randint(
                30, 50
            )  # Define a reasonable height for the arcs (higher = rounder, lower = flatter)
            start, end = (
                (0, 180) if arcToggle else (180, 0)
            )  # 1/2 Chance of reverse arc
            if randint(0, TogglingChance) == 0:
                arcToggle = not arcToggle
            draw_arc_with_condition(
                hasImperfectArcs,
                draw,
                lines_draw,
                mask_draw,
                [
                    float(0),
                    float(y + smallLineSize * j + offset),
                    float(pageWidth),
                    float(y + smallLineSize * j + offset + arc_height),
                ],
                start=start,
                end=end,
                fill=255,
                width=randint(1, 3) if j == 0 or j == smallLinesByLine - 1 else 5,
            )

    # Draw vertical lines with slight arcs to break regularity
    arcToggle = False
    for j in range(pageWidth // 60):  # Iterate over the page width
        vertical_x = j * 60
        arc_width = randint(30, 50)  # Define the width of the arc
        start, end = (90, 270) if arcToggle else (270, 90)  # 1/2 Chance of reverse arc
        if randint(0, TogglingChance) == 0:
            arcToggle = not arcToggle

        draw_arc_with_condition(
            hasImperfectArcs,
            draw,
            lines_draw,
            mask_draw,
            [
                float(vertical_x - arc_width),
                float(0),
                float(vertical_x + arc_width),
                float(pageHeight),
            ],
            start=start,
            end=end,
            fill=255,
            width=randint(1, 3),
        )
    linesImg.paste(PIL.Image.new(mode="L", size=(pageWidth, pageHeight)), mask=mask)
    page.paste(PIL.ImageOps.invert(linesImg), mask=linesImg)

    # Save the final page with lines to the pages directory
    page.save(f"{pages_dir}/{imageIndex}-page.png")
    if page.width > 2500 or page.height > 3500:
        print(
            f"[LineRemoverNN] [PageGenerator] WARNING: Page {imageIndex} is too large ({page.width}x{page.height}), consider reducing the size"
        )
    if split:
        blocks = split_into_blocks(ToTensor()(page))
        for i, block in enumerate(blocks):
            with blocks_index_l.get_lock():
                blocks_index_l.value += 1
                block_number = blocks_index_l.value - 1
            ToPILImage()(block).convert("L").save(
                os.path.join(pages_dir_blocks, f"{block_number}.png"),
                format="PNG",
            )


def init_worker(shared_values):
    global blocks_index, blocks_index_l
    blocks_index, blocks_index_l = shared_values


if __name__ == "__main__":
    # Parallelize page generation
    parser = argparse.ArgumentParser(
        prog="Page Generator",
        description="Generates realistic synthetic pages with and without lines for training",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output data folders",
        required=False,
        default="./",
    )
    parser.add_argument(
        "-p",
        "--pages",
        help="Number of pages to generate",
        type=int,
        required=False,
        default=1000,
    )
    parser.add_argument(
        "-s",
        "--split",
        help="Split directly the generated pages",
        action="store_true",
        required=False,
        default=False,
    )
    parser.add_argument(
        "-e",
        "--extended",
        help="Extended page generation, line skipping and random x",
        action="store_true",
        required=False,
        default=False,
    )
    parser.add_argument(
        "-b",
        "--blocks",
        help="Generate directly blocks",
        action="store_true",
        required=False,
        default=False,
    )
    args = parser.parse_args()
    if args.blocks:
        print(
            "[LineRemoverNN] [PageGenerator] FATAL: Blocks generation is broken, you may use split instead, aborting"
        )
        exit(1)
    print("[LineRemoverNN] [PageGenerator] Preparing for page generation")

    if not os.path.exists(args.output):
        os.mkdir(args.output)
        print("[LineRemoverNN] [PageGenerator] Created output directory ", args.output)

    data_dir = args.output

    pages_dir = os.path.join(data_dir, "generated-pages")
    nolines_dir = os.path.join(data_dir, "generated-nolines-pages")
    pages_dir_blocks = os.path.join(data_dir, "generated-pages-blocks")
    nolines_dir_blocks = os.path.join(data_dir, "generated-nolines-pages-blocks")
    json_dir = os.path.join(data_dir, "generated-words")

    for dir in [pages_dir, nolines_dir, pages_dir_blocks, nolines_dir_blocks, json_dir]:
        if not os.path.exists(dir):
            print("[LineRemoverNN] [PageGenerator] Creating directory ", dir)
            os.mkdir(dir)
        else:
            print(
                "[LineRemoverNN] [PageGenerator] WARNING: Removing existing data in dir ",
                dir,
            )
            shutil.rmtree(dir)
            os.mkdir(dir)

    num_pages = args.pages
    split = args.split  # Get the 'split' argument value
    extended = args.extended
    # Create a list of arguments for each page
    print(f"[LineRemoverNN] [PageGenerator] Generating {num_pages} pages")
    shared_values = (Value("i", 0), Value("i", 0))
    page_args = [
        (
            i,
            split,
            extended,
            (pages_dir, pages_dir_blocks, nolines_dir, nolines_dir_blocks, json_dir),
            ((0, 0) if not args.blocks else (512, 512)),
        )
        for i in range(num_pages)
    ]
    starttime = time.time_ns()
    with Pool(initializer=init_worker, initargs=(shared_values,)) as pool:
        list(tqdm.tqdm(pool.imap(make_page, page_args), total=num_pages))
    print(
        f"[LineRemoverNN] [PageGenerator] Finished generating {num_pages} pages in {(time.time_ns() - starttime) / 1e9} seconds"
    )
