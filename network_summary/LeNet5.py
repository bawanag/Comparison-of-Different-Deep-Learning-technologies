import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchsummary import summary
from torch.nn import functional as F
from pthflops import count_ops

class LeNet5(nn.Module):
   def __init__(self):
       super().__init__()
       self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
       self.conv2 = nn.Conv2d(6, 16, 5)
       self.ft = torch.nn.Flatten()
       self.fc1 = nn.Linear(46656, 120)
       self.fc2 = nn.Linear(120, 84)
       self.fc3 = nn.Linear(84, 10)
   def forward(self, x):
       x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
       x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
    #    x = x.view(-1, self.num_flat_features(x))
       x = self.ft(x)
       x = F.relu(self.fc1(x))
       x = F.relu(self.fc2(x))
       x = self.fc3(x)
       return x
   def num_flat_features(self, x):
       size = x.size()[1:]
       num_features = 1
       for s in size:
           num_features *= s
       return num_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
net = LeNet5().to(device)
net.eval()
summary(net,(1,224,224))


inp = torch.rand(1,1,224,224).to(device)
count_ops(net, inp)