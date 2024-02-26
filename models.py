from torch import nn
import torch.nn.functional as F
from hyperparameters import *
from torchvision.models import resnet50, ResNet50_Weights

class CNN(nn.Module):
    def __init__(self, l1_size=120, l2_size=84, kernel_size=3, dropout_rate=0.5, conv_1_size=6, conv_2_size=16, conv_3_size=120, conv_4_size=120):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, conv_1_size, kernel_size) # 1 input, conv_1_size output, kernel_size kernel size
        self.conv2 = nn.Conv2d(conv_1_size, conv_2_size, kernel_size)
        # self.conv3 = nn.Conv2d(conv_2_size, conv_3_size, kernel_size)
        # self.conv4 = nn.Conv2d(conv_3_size, conv_4_size, kernel_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.maxPool = nn.MaxPool2d(2, 2) # 2 kernel size, 2 stride, no padding
        self.fc1 = nn.Linear(conv_2_size * 5 * 5, l1_size) # 5x5 image dimension
        self.fc2 = nn.Linear(l1_size, l2_size)
        self.fc3 = nn.Linear(l2_size, 10)
        self.batch_norm1 = nn.BatchNorm2d(conv_1_size)
        self.batch_norm2 = nn.BatchNorm2d(conv_2_size)

    def forward(self, x):
        x = F.relu(self.maxPool(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.maxPool(self.conv2(x)))
        # x = self.dropout(x)
        # x = F.relu(self.maxPool(self.conv3(x)))
        x = torch.flatten(x, 1) 
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1) 

def create_cnn_model(l1_size: int,
                    l2_size: int,
                    kernel_size: int,
                    dropout_rate: float,
                    conv_1_size: int,
                    conv_2_size: int,
                    conv_3_size: int,
                    conv_4_size: int) -> CNN:
                        
    return CNN(l1_size, l2_size, kernel_size, dropout_rate, conv_1_size, conv_2_size, conv_3_size, conv_4_size).to(DEVICE)

resnet50Model = resnet50(weights=ResNet50_Weights.DEFAULT).to(DEVICE)

for param in resnet50Model.parameters():
    param.requires_grad = False

resnet50Model.fc = torch.nn.Linear(resnet50Model.fc.in_features, 10).to(DEVICE)