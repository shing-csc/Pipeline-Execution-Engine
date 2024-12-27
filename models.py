# models.py

import torch
import torch.nn as nn
from torch.distributed.rpc import RRef


# Key Configuration Parameters
num_classes = 1000  # Number of output classes
image_w = 128       # Image width for input
image_h = 128       # Image height for input
num_batches = 4     # Number of batches
batch_size = 64     # Number of samples per batch

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__() # Initialization of the parent class: nn.Module

        # features = A Neural Network Model, use a NN model to do feature extraction
        #            Below is the code for define the structure inside the features NN 
        self.features = nn.Sequential(
            # Feature extraction layers
            nn.Conv2d(3, 64, kernel_size=3, padding=1),    # 0
            nn.ReLU(inplace=True),                         # 1
            nn.Conv2d(64, 64, kernel_size=3, padding=1),   # 2
            nn.ReLU(inplace=True),                         # 3
            nn.MaxPool2d(kernel_size=2, stride=2),         # 4

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 5
            nn.ReLU(inplace=True),                         # 6
            nn.Conv2d(128, 128, kernel_size=3, padding=1), # 7
            nn.ReLU(inplace=True),                         # 8
            nn.MaxPool2d(kernel_size=2, stride=2),         # 9

            nn.Conv2d(128, 256, kernel_size=3, padding=1), #10
            nn.ReLU(inplace=True),                         #11
            nn.Conv2d(256, 256, kernel_size=3, padding=1), #12
            nn.ReLU(inplace=True),                         #13
            nn.Conv2d(256, 256, kernel_size=3, padding=1), #14
            nn.ReLU(inplace=True),                         #15
            nn.Conv2d(256, 256, kernel_size=3, padding=1), #16
            nn.ReLU(inplace=True),                         #17
            nn.MaxPool2d(kernel_size=2, stride=2),         #18

            nn.Conv2d(256, 512, kernel_size=3, padding=1), #19
            nn.ReLU(inplace=True),                         #20
            nn.Conv2d(512, 512, kernel_size=3, padding=1), #21
            nn.ReLU(inplace=True),                         #22
            nn.Conv2d(512, 512, kernel_size=3, padding=1), #23
            nn.ReLU(inplace=True),                         #24
            nn.Conv2d(512, 512, kernel_size=3, padding=1), #25
            nn.ReLU(inplace=True),                         #26
            nn.MaxPool2d(kernel_size=2, stride=2),         #27

            nn.Conv2d(512, 512, kernel_size=3, padding=1), #28
            nn.ReLU(inplace=True),                         #29
            nn.Conv2d(512, 512, kernel_size=3, padding=1), #30
            nn.ReLU(inplace=True),                         #31
            nn.Conv2d(512, 512, kernel_size=3, padding=1), #32
            nn.ReLU(inplace=True),                         #33
            nn.Conv2d(512, 512, kernel_size=3, padding=1), #34
            nn.ReLU(inplace=True),                         #35
            nn.MaxPool2d(kernel_size=2, stride=2),         #36
        )
        
        # classifier = A Neural Network Model, used a NN model to do classification of images
        self.classifier = nn.Sequential(
            # Fully connected layers for classification
            nn.Linear(512 * 4 * 4, 4096),                  #37
            nn.ReLU(inplace=True),                         #38
            nn.Dropout(),                                  #39
            nn.Linear(4096, 4096),                         #40
            nn.ReLU(inplace=True),                         #41
            nn.Dropout(),                                  #42
            nn.Linear(4096, num_classes),                  #43
        )

    def forward(self, x):

        # Step 1: Pass input to the "features" NN model (specifically its feature layers)
        x = self.features(x)  

        # Step 2: Flatten the 
        #       4-D feature map: shape = (batch size, Number of feature maps, height, width)), to 
        #       2D-tensor:       shape = (batch size, features)  
        #       * Note that flattens didn't change the number of data, just the dimension   
        x = x.view(x.size(0), -1)
        
        # Step 3: Passing the flattened features into the "classifier" NN model 
        x = self.classifier(x)
        
        return x

class Partition(nn.Module):
    def __init__(self, modules):
        super(Partition, self).__init__()
        self.modules = modules

    def forward(self, x_rref):
        x = x_rref.to_here().to("cpu")  # Fetch the input tensor from the remote reference and move to CPU
        for i, module in enumerate(self.modules):
            x = module(x)
            # Flatten the tensor after MaxPool2d if needed for Linear layers
            if isinstance(module, nn.MaxPool2d):
                # Check if next module is Linear, then flatten
                if i + 1 < len(self.modules) and isinstance(self.modules[i + 1], nn.Linear):
                    x = x.view(x.size(0), -1)  # Flatten only if next layer is Linear
        return x

    def parameter_rrefs(self):
        # Return remote references for model parameters
        return [RRef(p) for p in self.parameters()]
