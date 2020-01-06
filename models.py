## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 4)
        self.conv3 = nn.Conv2d(64, 128, 3)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 25 * 25, 1000)
        self.fc2 = nn.Linear(1000, 136)
        
        # MaxPooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout layers
        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.2)
        self.drop3 = nn.Dropout(0.3)
        self.drop4 = nn.Dropout(0.4)

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model

        # Conv2d-ReLU-MaxPool2d-Dropout blocks
        # 224 -> 110 -> 53 -> 25
        x = self.drop1(self.pool(F.relu(self.conv1(x))))
        x = self.drop2(self.pool(F.relu(self.conv2(x))))
        x = self.drop3(self.pool(F.relu(self.conv3(x))))
        
        # Flattening
        x = x.view(x.shape[0], -1)
        
        # fully connected layers
        x = self.drop4(F.relu(self.fc1(x)))
        x = self.fc2(x)

        # a modified x, having gone through all the layers of your model, will be returned
        return x