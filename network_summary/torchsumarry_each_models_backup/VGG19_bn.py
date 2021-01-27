import torch
import torch.nn as nn
from torchsummary import summary
from torchvision import models
from pthflops import count_ops
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = models.vgg19_bn().to(device)

summary(net,(3,224,224))
inp = torch.rand(1,3,224,224).to(device)
count_ops(net, inp)
