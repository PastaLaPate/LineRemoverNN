import PIL.Image
import PIL.ImageDraw
import numpy as np
from random import randint
import tqdm
from functools import lru_cache
from multiprocessing import Pool

# Load word metadata
words = []
with open('./words.txt', encoding='UTF-8', mode='r') as words_file:
    for line in words_file:
        line = line.rstrip()
        if line.startswith('#') or len(line.split(' ')) > 9:
            continue
        filename, segmentation, grayScale, x, y, w, h, typ, transcript = line.split(' ')
        if segmentation == 'err':
            continue
        fnamesplit = filename.split('-')
        path = f"./words/{fnamesplit[0]}/{'-'.join(fnamesplit[:2])}/{filename}.png"
        words.append((path, (int(x), int(y), int(w), int(h)), transcript, int(grayScale)))

pages_dir = './generated-pages/'
nolines_dir = './generated-nolines-pages/'

# Transparent background function
def makeTransparentBG(img):
    """Make the white background of an image transparent."""
    img_data = np.array(img)
    avg = img_data[..., :3].mean(axis=2)
    mask = avg <= 200
    img_data[..., 3] = mask * 255
    return PIL.Image.fromarray(img_data, 'RGBA')

# Cache images with a size limit to reduce repeated loads
@lru_cache(maxsize=1000)
def load_image(path):
    """Load and preprocess a word image."""
    try:
        img = PIL.Image.open(path).convert('RGBA')
        return makeTransparentBG(img)
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None

def make_page(imageIndex):
    """Generate a single page with random words."""
    page = PIL.Image.new(mode='RGBA', size=(2480, 3508), color=(255, 255, 255))
    draw = PIL.ImageDraw.Draw(page)
    x = 0
    y = 0
    lineSize = randint(120, 200)
    smallLineSize = lineSize / 4
    lines = []
    gS = 255

    # Generate content
    for i in range(len(words)):
        word = words[i % len(words)]
        path, box, transcript, wgS = word
        if wgS < gS:
            gS = wgS
        _, _, w, h = box
        if x + w >= 2480:
            x = 0
            lines.append(y)
            y += lineSize
        if y >= 3508 - lineSize:
            break
        # Load word image on demand
        wordimg = load_image(path)
        if wordimg is None:
            continue
        # Paste word image onto page
        page.paste(wordimg, (x, y - h + randint(-20, 20)), wordimg)
        x += w + randint(-10, 30)

    page.save(f"{nolines_dir}/{imageIndex}-page.png")
    # Draw lines
    for y in lines:
        for j in range(4):
            draw.line((0, y + smallLineSize * j, 2480, y + smallLineSize * j), (gS - 40, gS - 40, gS - 40), 3)
    for j in range(3508 // 60):
        draw.line((j * 60, 0, j * 60, 3508), (gS - 40, gS - 40, gS - 40), 2)

    page.save(f"{pages_dir}/{imageIndex}-page.png")

if __name__ == '__main__':
    # Parallelize page generation
    num_pages = 200
    with Pool() as pool:
        list(tqdm.tqdm(pool.imap(make_page, range(num_pages)), total=num_pages))
