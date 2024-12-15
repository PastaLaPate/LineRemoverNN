import PIL.Image
import PIL.ImageDraw
import numpy as np
from random import randint
import tqdm
from functools import lru_cache
from multiprocessing import Pool
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


# Cache images with a size limit to reduce repeated loads
@lru_cache(maxsize=1000)
def load_image(path):
    """Load and preprocess a word image."""
    try:
        img = PIL.Image.open(path).convert("RGBA")
        return makeTransparentBG(img)
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None


def make_page(imageIndex):
    """Generate a single page with random words."""

    # Create a blank page with white background
    page = PIL.Image.new(mode="RGBA", size=(2480, 3508), color=(255, 255, 255))
    draw = PIL.ImageDraw.Draw(page)

    # Initialize variables for positioning
    x = 30
    y = 300
    lineSize = randint(120, 200)  # Random line height for text
    smallLinesByLine = randint(4, 7)
    smallLineSize = (
        lineSize / smallLinesByLine
    )  # Distance between small horizontal lines
    lines = []  # List to store y-coordinates of lines
    wordsL = []  # List to store metadata about the words on the page
    gS = 255  # Grayscale value (used for line color)

    # Generate content by iterating through the words list
    for i in range(len(words)):
        word = words[i % len(words)]  # Cycle through the words list
        path, box, transcript, wgS = word  # Unpack word details

        # Update grayscale to the darkest word color so far
        if wgS < gS:
            gS = wgS

        # Extract dimensions of the word box
        _, _, w, h = box

        # Move to a new line if the word doesn't fit in the current row
        if x + w >= 2480:
            x = 30
            lines.append(y)  # Add current y-coordinate to lines list
            y += lineSize

        # Stop adding words if the page height is exceeded
        if y >= 3508 - lineSize:
            break

        # Load the word image
        wordimg = load_image(path)
        if wordimg is None:
            continue

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
            draw.arc(
                [
                    float(0),
                    float(y + smallLineSize * j + offset),
                    float(2480),
                    float(y + smallLineSize * j + offset + arc_height),
                ],
                start=start,
                end=end,
                fill=(gS - 40, gS - 40, gS - 40),
                width=randint(1, 6),
            )

    # Draw vertical lines with slight arcs to break regularity
    arcToggle = False
    for j in range(2480 // 60):  # Iterate over the page width
        vertical_x = j * 60
        arc_width = randint(30, 50)  # Define the width of the arc
        start, end = (90, 270) if arcToggle else (270, 90)  # 1/2 Chance of reverse arc
        if randint(0, TogglingChance) == 0:
            arcToggle = not arcToggle
        draw.arc(
            [
                float(vertical_x - arc_width),
                float(0),
                float(vertical_x + arc_width),
                float(3508),
            ],
            start=start,
            end=end,
            fill=(gS - 40, gS - 40, gS - 40),
            width=randint(1, 6),
        )

    # Save the final page with lines to the pages directory
    page.save(f"{pages_dir}/{imageIndex}-page.png")


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
    args = parser.parse_args()

    if not os.path.exists(args.output):
        raise ValueError("Output folder does not exist")
    data_dir = args.output
    pages_dir = os.path.join(data_dir, "generated-pages")
    nolines_dir = os.path.join(data_dir, "generated-nolines-pages")
    json_dir = os.path.join(data_dir, "generated-words")

    if not (
        os.path.exists(pages_dir)
        and os.path.exists(nolines_dir)
        and os.path.exists(json_dir)
    ):
        os.mkdir(pages_dir)
        os.mkdir(nolines_dir)
        os.mkdir(json_dir)
    num_pages = args.pages
    with Pool() as pool:
        list(tqdm.tqdm(pool.imap(make_page, range(num_pages)), total=num_pages))
