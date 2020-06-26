import cv2 as cv
import os
import numpy as np
from tqdm import tqdm

ORG_DIR = "org_img"
INPUT_DATA_DIR = "dataset/input"
OUTPUT_DATA_DIR = "dataset/output"
INPUT_SIZE = 572

def reshape(img):
    height = img.shape[0]
    width = img.shape[1]
    if height > width:
        scale = INPUT_SIZE / height
    else:
        scale = INPUT_SIZE / width
    size = (int(height * scale), int(width * scale))
    image = cv.resize(img, size)
    top = int((INPUT_SIZE-image.shape[0])/2)
    left = int((INPUT_SIZE-image.shape[1])/2)
    image = cv.copyMakeBorder(image, top, INPUT_SIZE-image.shape[0]-top, left, INPUT_SIZE-image.shape[1]-left, cv.BORDER_CONSTANT)
    return image

i = 0
for file in tqdm(os.listdir(ORG_DIR)):
    file = os.path.join(ORG_DIR, file)
    try:
        img = cv.imread(file, cv.IMREAD_GRAYSCALE)
        img = img.astype(np.uint8)
        image = reshape(img)
        name = file[len(ORG_DIR) + 1:-4]
        name = name.zfill(7)
        path = os.path.join(OUTPUT_DATA_DIR, name + ".jpg")
        cv.imwrite(path, image)  # ORIGINAL IMAGE IN GRAYSCALE

        noise = (np.random.random(img.shape) * 255).astype(np.uint8)
        noisy = cv.addWeighted(img, 0.8, noise, 0.2, 0)
        noisy = reshape(noisy)
        path = os.path.join(INPUT_DATA_DIR, name + "_1.jpg")
        cv.imwrite(path, noisy) # NOISY IMAGE WITH 8:2 RATIO

        noise = (np.random.random(img.shape) * 255).astype(np.uint8)
        noisy = cv.addWeighted(img, 0.6, noise, 0.4, 0)
        noisy = reshape(noisy)
        path = os.path.join(INPUT_DATA_DIR, name + "_2.jpg")
        cv.imwrite(path, noisy)  # NOISY IMAGE WITH 6:4 RATIO
        i += 1
    except Exception:
        pass
print("Total = ", i)