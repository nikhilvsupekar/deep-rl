import os

import torch
import torch.nn as nn
import torchvision.models as models

class Model:
    
    def __init__(self):
        self.model = models.resnet18(pretrained = False)
        self.model.conv1 = nn.Conv2d(12, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(512, 3)

    def load(self, file_name):
        saved_dict = torch.load(file_name, map_location = torch.device('cpu'))
        self.model.load_state_dict(saved_dict)

    def save(self, file_name):
        torch.save(
            self.model.state_dict(), 
            os.path.join(file_name)
        )
