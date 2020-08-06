import torch
import numpy as np
import os
from tqdm import tqdm
import cv2 as cv
import model
import pickle

FILE = "dataset/input"
mdl_file = "model/UNet_mdl5.pickle"

# if torch.cuda.is_available():
#     device = torch.device("cuda:0")
#     print("Running on GPU")
# else:
#     device = torch.device("cpu")
#     print("Running on CPU")
device = torch.device("cpu")
files =[]

if os.path.isdir(FILE):
    for fi in os.listdir(FILE):
        files.append(os.path.join(FILE, fi))
else:
    files.append(FILE)

# net = model.UNet()
file = open(mdl_file, "rb")
net = pickle.load(file)
file.close()
net.to(device)

for f_img in files:
    with torch.no_grad():
        img = cv.imread(f_img, cv.IMREAD_GRAYSCALE)
        cv.imshow("INPUT", img)
        img = torch.Tensor([np.array((img))]).view(-1, 1, net.INPUT_SIZE, net.INPUT_SIZE)
        img = img / 255
        img = img.to(device)
        out = net(img)
        out = out.to('cpu')
        out = out.view(net.OUTPUT_SIZE, net.OUTPUT_SIZE).numpy()
        cv.imshow("OUTPUT", out)
        cv.waitKey(600)
