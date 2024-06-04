import torch.nn as nn
import torch


########################
# Define the CNN model
class Conv1DModel(nn.Module):
    def __init__(self, input_channels, kernel_size=10):
        super(Conv1DModel, self).__init__()
        self.conv1d = nn.Conv1d(input_channels, 64, kernel_size=10, padding='same')
        self.conv1d_2 = nn.Conv1d(64, 32, kernel_size=kernel_size // 2, padding='same' , dilation=2)
        self.conv1d_3 = nn.Conv1d(32, 8, kernel_size=2, padding='same')
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2000, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x):
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.conv1d_2(x)
        x = self.relu(x)
        x = self.conv1d_3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

