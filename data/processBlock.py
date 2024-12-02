from IAM import split_into_blocks
import os
import cv2
from torchvision.transforms import ToTensor, ToPILImage
from multiprocessing import Pool
from tqdm import tqdm


# Process a single file
def process_file(args):
    file, folderPath, newFolderPath, idx = args
    if file.endswith('.png'):
        loadedImg = cv2.imread(os.path.join(folderPath, file))
        blocks = split_into_blocks(ToTensor()(loadedImg))
        for i, block in enumerate(blocks):
            ToPILImage()(block).save(os.path.join(newFolderPath, f"{idx * len(blocks) + i}.png"))


# Process all files in a folder
def process_folder(folderPath, newFolderPath):
    os.makedirs(newFolderPath, exist_ok=True)
    files = [file for file in os.listdir(folderPath) if file.endswith('.png')]
    
    # Prepare arguments for multiprocessing
    args = [(file, folderPath, newFolderPath, idx) for idx, file in enumerate(files)]
    
    # Use a multiprocessing pool to process files
    with Pool() as pool:
        list(tqdm(pool.imap(process_file, args), total=len(args)))


if __name__ == '__main__':
    # Process folders with multiprocessing
    process_folder('./generated-pages', './generated-pages-blocks')
    process_folder('./generated-nolines-pages', './generated-nolines-pages-blocks')
