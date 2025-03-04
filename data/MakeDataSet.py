import PIL.Image
import PIL.ImageDraw
import PIL.ImageOps
import numpy as np
from random import randint, uniform as randfloat
import tqdm
from functools import lru_cache
from multiprocessing import Pool
from IAM import split_into_blocks
from torchvision.transforms import ToTensor, ToPILImage
from random import randint, uniform
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


def load_image(path):
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

def make_page(args):
    """Generate a single page with random words."""
    imageIndex, split, extended, dirs = args
    pages_dir, pages_dir_blocks, nolines_dir, nolines_dir_blocks, json_dir = dirs
    randomPageSize = lambda: 3000 if not extended else randint(1000, 4000)
    pageWidth, pageHeight = randomPageSize(), randomPageSize()
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
    y = 300
    lineSize = randint(120, 200)  # Random line height for text
    smallLinesByLine = randint(4, 7)
    smallLineSize = (
        lineSize / smallLinesByLine
    )  # Distance between small horizontal lines
    lines = []  # List to store y-coordinates of lines
    wordsL = []  # List to store metadata about the words on the page
    gS = 255  # Grayscale value (used for line color)
    hasImperfectArcs = bool(randint(0, 1))  # Randomly decide whether to add arcs
    skipLineProb = (
        1 / 4 if extended else 0
    )  # Probability of skipping one or multiple lines
    stopX = pageWidth - randX()

    # Generate content by iterating through the words list
    i = randint(0, len(words))
    while True:
        word = words[i % len(words)]  # Cycle through the words list
        path, box, transcript, wgS = word  # Unpack word details

        # Update grayscale to the darkest word color so far
        if wgS < gS:
            gS = wgS

        # Extract dimensions of the word box
        _, _, w, h = box

        # Load the word image
        wordimg = load_image(path)
        if wordimg is None:
            words.pop(i % len(words))
            continue
        i += 1

        # Move to a new line if the word doesn't fit in the current row
        if x + w >= stopX:
            x = randX()
            stopX = pageWidth - randX()
            lines.append(y)  # Add current y-coordinate to lines list
            y += lineSize if randint(0, 1) < skipLineProb else lineSize * randint(1, 2)

        # Stop adding words if the page height is exceeded
        if y >= pageHeight - lineSize:
            break


        # Random vertical adjustment for the word placement
        rn = randint(-20, 20)

        # Add metadata about the word to wordsL list
        wordsL.append({"text": transcript, "x": x, "y": y - h + rn, "w": w, "h": h})

        # Paste the word image onto the page
        page.paste(wordimg, (x, y - h + rn), wordimg)

        # Update x-coordinate for the next word, with a random gap
        x += w + randint(-10, 30)

    # Save the page without lines to a separate directory
    page.save(f"{nolines_dir}/{imageIndex}-page.png")
    if split:
        blocks = split_into_blocks(ToTensor()(page))
        for i, block in enumerate(blocks):
            ToPILImage()(block).save(
                os.path.join(nolines_dir_blocks, f"{imageIndex * len(blocks) + i}.png")
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
                width=randint(1, 3),
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

    if split:
        blocks = split_into_blocks(ToTensor()(page))
        for i, block in enumerate(blocks):
            ToPILImage()(block).save(
                os.path.join(pages_dir_blocks, f"{imageIndex * len(blocks) + i}.png")
            )


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
    args = parser.parse_args()

    if not os.path.exists(args.output):
        raise ValueError("Output folder does not exist")
    
    data_dir = args.output
    
    pages_dir = os.path.join(data_dir, "generated-pages")
    nolines_dir = os.path.join(data_dir, "generated-nolines-pages")
    pages_dir_blocks = os.path.join(data_dir, "generated-pages-blocks")
    nolines_dir_blocks = os.path.join(data_dir, "generated-nolines-pages-blocks")
    json_dir = os.path.join(data_dir, "generated-words")

    for dir in [pages_dir, nolines_dir, pages_dir_blocks, nolines_dir_blocks, json_dir]:
        if not os.path.exists(dir):
            os.mkdir(dir)
    
    num_pages = args.pages
    split = args.split  # Get the 'split' argument value
    extended = args.extended
    # Create a list of arguments for each page
    print(f"[LineRemoverNN] [PageGenerator] Generating {num_pages} pages")
    page_args = [(i, split, extended, (pages_dir, pages_dir_blocks, nolines_dir, nolines_dir_blocks, json_dir)) for i in range(num_pages)]
    starttime = time.time_ns()
    with Pool() as pool:
        list(tqdm.tqdm(pool.imap(make_page, page_args), total=num_pages))
    print(
        f"[LineRemoverNN] [PageGenerator] Finished generating {num_pages} pages in {(time.time_ns() - starttime) / 1e9} seconds"
    )
