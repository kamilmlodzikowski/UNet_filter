import torch
import numpy as np
import os
from tqdm import tqdm
import cv2 as cv
import model
import pickle

DATA_MAX_SIZE = 1000

FILE = "dataset/input"
mdl_file = "model/UNet_mdl105.pickle"

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

#for f_img in files[DATA_MAX_SIZE:DATA_MAX_SIZE*2]:
for f_img in files[:DATA_MAX_SIZE]:
    with torch.no_grad():
        img = cv.imread(f_img, cv.IMREAD_COLOR)
        cv.imshow("INPUT", img)
        # print(img.shape)
        # print(220*220*3)
        # cv.waitKey(0)
        img = torch.Tensor([np.array((img))]).view(-1, 3, net.INPUT_SIZE, net.INPUT_SIZE)
        # print(img.shape)
        img = img / 255
        img = img.to(device)
        out = net(img)
        out = out.to('cpu')
        out = out.view(net.OUTPUT_SIZE, net.OUTPUT_SIZE, 3).numpy()
        cv.imshow("OUTPUT", out)
        cv.waitKey(500)
