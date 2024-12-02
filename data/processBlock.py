from IAM import split_into_blocks
import os
import cv2
from torchvision.transforms import ToTensor, ToPILImage
import PIL.Image as PIL
from multiprocessing import Pool
from tqdm import tqdm

# All images should be the same size
def process_folder(folderPath, newFolderPath):
    files = os.listdir(folderPath)
    # [:10] for testing
    for _, file in enumerate(files[:10]):
        if file.ends_with('.png'):
            loadedImg = cv2.imread(os.path.join(folderPath, file))
            blocks = split_into_blocks(ToTensor()(loadedImg))
            for i, block in enumerate(blocks):
                ToPILImage()(block).save(os.path.join(newFolderPath, f"{_*len(blocks)+i}"))
                

if __name__ == '__main__':
    with Pool() as pool:
        process_folder('./generated-pages', './generated-pages-blocks')
        process_folder('./generated-nolines-pages', './generated-nolines-pages-blocks')