from IAM import split_into_blocks
import os
import cv2
from torchvision.transforms import ToTensor, ToPILImage
from multiprocessing import Pool
from tqdm import tqdm
import argparse


# Process a single file
def process_file(args):
    file, folderPath, newFolderPath, idx = args
    if file.endswith(".png"):
        loadedImg = cv2.imread(os.path.join(folderPath, file))
        blocks = split_into_blocks(ToTensor()(loadedImg))
        for i, block in enumerate(blocks):
            ToPILImage()(block).save(
                os.path.join(newFolderPath, f"{idx * len(blocks) + i}.png")
            )


# Process all files in a folder
def process_folder(folderPath, newFolderPath):
    os.makedirs(newFolderPath, exist_ok=True)
    files = [file for file in os.listdir(folderPath) if file.endswith(".png")]

    # Prepare arguments for multiprocessing
    args = [(file, folderPath, newFolderPath, idx) for idx, file in enumerate(files)]

    # Use a multiprocessing pool to process files
    with Pool() as pool:
        list(tqdm(pool.imap(process_file, args), total=len(args)))


if __name__ == "__main__":
    # Process folders with multiprocessing
    # pages_dir = '/mnt/c/users/alexa/DatasetData/generated-pages/'
    # nolines_dir = '/mnt/c/users/alexa/DatasetData/generated-nolines-pages/'
    # pages_blocks_dir = '/mnt/c/users/alexa/DatasetData/generated-pages-blocks/'
    # nolines_blocks_dir = '/mnt/c/users/alexa/DatasetData/generated-nolines-pages-blocks/'
    parser = argparse.ArgumentParser(
        prog="Page splitter", description="Split pages into blocks of 512x512"
    )
    parser.add_argument("-d", "--dir", default="./")
    args = parser.parse_args()
    data_dir = args.dir
    nolines_dir = os.path.join(data_dir, "generated-nolines-pages")
    pages_dir = os.path.join(data_dir, "generated-pages")
    nolines_blocks_dir = os.path.join(data_dir, "generated-nolines-pages-blocks")
    pages_blocks_dir = os.path.join(data_dir, "generated-pages-blocks")
    if not (os.path.exists(nolines_blocks_dir) or os.path.exists(pages_blocks_dir)):
        os.mkdir(pages_blocks_dir)
        os.mkdir(nolines_blocks_dir)
    process_folder(pages_dir, pages_blocks_dir)
    process_folder(nolines_dir, nolines_blocks_dir)
