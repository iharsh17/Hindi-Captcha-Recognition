import pandas as pd
import numpy as np
import cv2
import os
import json

import torch
from torch import nn
import torch.nn.functional as F


class HindiCharNet(nn.Module):
    """
    For using forward/__call__ function:
    Input: Tensor of shape (Batch, 1, 32, 32) ie (Batch, Channels, X, Y)
    Output: logits as a Tensor of shape (Batch, NUM_CLASSES)
    """

    def __init__(self, NUM_CLASSES, NUM_CHANNELS=1):
        super(HindiCharNet, self).__init__()

        self.conv1 = nn.Conv2d(NUM_CHANNELS, 100, 5)
        self.conv1_bn = nn.BatchNorm2d(100)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(100, 150, 3)
        self.conv2_bn = nn.BatchNorm2d(150)
        self.conv3 = nn.Conv2d(150, 250, 1)
        self.conv3_bn = nn.BatchNorm2d(250)
        self.fc1 = nn.Linear(250 * 3 * 3, 350)
        self.fc1_bn = nn.BatchNorm1d(350)
        self.fc2 = nn.Linear(350, NUM_CLASSES)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):

        x = self.pool(F.elu(self.conv1(x)))
        x = self.dropout(self.conv1_bn(x))
        x = self.pool(F.elu(self.conv2(x)))
        x = self.dropout(self.conv2_bn(x))
        x = self.pool(F.elu(self.conv3(x)))
        x = self.dropout(self.conv3_bn(x))
        # x.shape = (BATCH, 250, 3, 3)
        x = x.view(-1, 250 * 3 * 3)
        x = F.elu(self.fc1(x))
        x = self.dropout(self.fc1_bn(x))
        x = self.fc2(x)
        return x


class ClassifierInference:
    def __init__(self, model_state_path, labels_map_json_path):

        with open(labels_map_json_path) as f:
            self.labels2devchar = json.load(f)
        NUM_CLASSES = len(self.labels2devchar)

        self.model = HindiCharNet(NUM_CLASSES)
        self.model.load_state_dict(torch.load(model_state_path))
        self.model.eval()

    def predict(self, image):
        """
        No batching, image input must be a single image Grayscale image of shape (X,Y)
        returns: single character
        """

        # image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (32, 32))
        img_tensor = torch.tensor(image, dtype=torch.float)
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0) / 255.0

        with torch.no_grad():
            logits = self.model(img_tensor)
        label = logits.softmax(-1).argmax(-1)[0].item()
        return self.labels2devchar[str(label)]
