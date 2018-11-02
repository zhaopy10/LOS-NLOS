import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, K=288, Tx=8, Rx=2):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=Rx*2, out_channels=32, 
                               kernel_size=3, stride=(1,2), padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, 
                               kernel_size=3, stride=(1,2), padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, 
                               kernel_size=3, stride=(1,2), padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, 
                               kernel_size=3, stride=(1,2), padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, 
                               kernel_size=3, stride=(1,2), padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, 
                               kernel_size=3, stride=(2,2), padding=1)
        
      
        self.fc = nn.Linear(256, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        
        K_dim_size = x.shape[2]
        Tx_dim_size = x.shape[3]
        x = nn.MaxPool2d(kernel_size=(K_dim_size, Tx_dim_size))(x)
        
        x = x.view(-1, 256)
        x = self.fc(x)
        x = self.activation(x)
        x = x.view(-1)
        
        return x
