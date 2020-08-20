import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import cv2 as cv
import model
import pickle

REBUILD_DATA = False
HARDER_DATA = False

LOAD_MODEL = False
MODEL_FILE = "model/UNet_mdl179.pickle"

IN_DIR = "dataset/input"
OUT_DIR = "dataset/output"
DATA_MAX_SIZE = 1000

BATCH_SIZE = 8
EPOCHS = 500


def load_data(rebuild=REBUILD_DATA):
    global training_dataY, training_dataX

    if rebuild:
        print("LOADING RAW DATA")
        create_training_data(IN_DIR, OUT_DIR)

        print("PROCESSING RAW DATA")

        training_dataX = torch.Tensor([i for i in training_dataX]).view(-1, net.INPUT_SIZE, net.INPUT_SIZE)
        training_dataX = training_dataX / 255.0

        training_dataY = torch.Tensor([i for i in training_dataY]).view(-1, net.OUTPUT_SIZE, net.OUTPUT_SIZE)
        training_dataY = training_dataY / 255.0

        print("FINISHED PROCESSING RAW DATA")

        print("SAVING DATA")
        file = open("training_dataX.pickle", 'wb')
        pickle.dump(training_dataX, file)
        file.close()
        file = open("training_dataY.pickle", 'wb')
        pickle.dump(training_dataY, file)
        file.close()
        # np.save("training_dataX.npy", training_dataX)
        # np.save("training_dataY.npy", training_dataY)
        print("FINISHED SAVING DATA")

    else:
        print("LOADING DATA")
        file = open("training_dataX.pickle", 'rb')
        training_dataX = pickle.load(file)
        file.close()
        file = open("training_dataY.pickle", 'rb')
        training_dataY = pickle.load(file)
        file.close()
        # training_dataX = np.load("training_dataX.npy", allow_pickle=True)
        # training_dataY = np.load("training_dataY.npy", allow_pickle=True)
        print("FINISHED LOADING DATA")


def create_training_data(dir_input, dir_output):
    global training_dataX, training_dataY
    training_dataX = []
    training_dataY = []
    for file in tqdm(os.listdir(dir_output)[:DATA_MAX_SIZE]):
        try:
            out_path = os.path.join(dir_output, file)
            out_img = cv.imread(out_path, cv.IMREAD_GRAYSCALE)
            in_file = file[:-4]+'_1.jpg'
            in_path = os.path.join(dir_input, in_file)
            in_img = cv.imread(in_path, cv.IMREAD_GRAYSCALE)
            training_dataX.append(np.array(in_img))
            training_dataY.append(np.array(out_img))

            if HARDER_DATA:
                in_file = file[:-4] + '_2.jpg'
                in_path = os.path.join(dir_input, in_file)
                in_img = cv.imread(in_path, cv.IMREAD_GRAYSCALE)
                training_dataX.append(np.array(in_img))
                training_dataY.append(np.array(out_img))

        except Exception:
            pass
    # np.random.shuffle(training_data)
    return training_dataX, training_dataY


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on GPU")
else:
    device = torch.device("cpu")
    print("Running on CPU")

# CREATING MODEL
if LOAD_MODEL:
    mdl_file = open(MODEL_FILE, "rb")
    net = pickle.load(mdl_file)
    mdl_file.close()
else:
    net = model.UNet().to(device)

net = net.to(device)
# net.loss_function = nn.MSELoss()

training_dataX = torch.Tensor()
training_dataY = torch.Tensor()

load_data(rebuild=REBUILD_DATA)

# training_dataX.to(device)
# training_dataY.to(device)

for epoch in range(EPOCHS):
    print("Epoch: ", epoch)
    for i in tqdm(range(0, len(training_dataX), BATCH_SIZE)):
        batch_X = training_dataX[i:i + BATCH_SIZE].view(-1, 1, net.INPUT_SIZE, net.INPUT_SIZE)
        batch_X = batch_X.to(device)

        batch_Y = training_dataY[i:i + BATCH_SIZE].view(-1, 1, net.OUTPUT_SIZE, net.OUTPUT_SIZE)

        net.zero_grad()
        outputs = net(batch_X)
        # print("OUT: ", outputs.shape)
        # print("Y:   " ,batch_Y.shape)
        del batch_X
        torch.cuda.empty_cache()

        batch_Y = batch_Y.to(device)
        loss = net.loss_function(outputs, batch_Y)
        loss.backward()
        net.optimizer.step()
        del batch_Y
        torch.cuda.empty_cache()
    
    if epoch % 5 == 0:
        print("SAVING MODEL FOR EPOCH ", epoch)
        mdl_name = net.MODEL_NAME + str(epoch)+".pickle"
        file = open(mdl_name, "wb")
        pickle.dump(net, file)
        file.close()
        print("MODEL SAVED")

    print("\nLOSS: ", loss, "\n")


