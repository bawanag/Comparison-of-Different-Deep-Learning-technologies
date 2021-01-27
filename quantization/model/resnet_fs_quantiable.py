import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
import torch

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, sketch_rate=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, int(planes * sketch_rate), kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(planes * sketch_rate))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(int(planes * sketch_rate), planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.add_relu = torch.nn.quantized.FloatFunctional()

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        x = self.downsample(x)
        out = self.add_relu.add_relu(out, x) 
        return out

    def fuse_model(self):
        torch.quantization.fuse_modules(self, [['conv1', 'bn1', 'relu'],
                                               ['conv2', 'bn2']], inplace=True)
        if self.downsample:
            torch.quantization.fuse_modules(self.downsample, ['0', '1'], inplace=True)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, sketch_rate=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, int(planes * sketch_rate), kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(planes * sketch_rate))
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(int(planes * sketch_rate), int(planes * sketch_rate), kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(int(planes * sketch_rate))
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = nn.Conv2d(int(planes * sketch_rate), self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.downsample = nn.Sequential()
        self.skip_add_relu = nn.quantized.FloatFunctional()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        x = self.downsample(x)
        # out = F.relu(out)
        out = self.skip_add_relu.add_relu(out, x) 
        return out

    def fuse_model(self):
        fuse_modules(self, [['conv1', 'bn1', 'relu1'],
                            ['conv2', 'bn2', 'relu2'],
                            ['conv3', 'bn3']], inplace=True)
        if self.downsample:
            torch.quantization.fuse_modules(self.downsample, ['0', '1'], inplace=True)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, sketch_rate=None, start_conv=1):
        super(ResNet, self).__init__()
        self.in_planes = 64
        if sketch_rate is None:
            self.sketch_rate = [1] * sum(num_blocks)
        else:
            self.sketch_rate = sketch_rate
        self.start_conv = start_conv
        self.current_conv = 0

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.current_conv += 1

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.Sequential(nn.AvgPool2d(7))
        self.fc = nn.Linear(512*block.expansion, num_classes)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,
                                sketch_rate=self.sketch_rate[self.current_conv - self.start_conv]
                                if self.current_conv >= self.start_conv else 1.0))
            self.current_conv += 1
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.quant(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.dequant(out)
        return out

    def fuse_model(self):
        r"""Fuse conv/bn/relu modules in resnet models

        Fuse conv+bn+relu/ Conv+relu/conv+Bn modules to prepare for quantization.
        Model is modified in place.  Note that this operation does not change numerics
        and the model after modification is in floating point
        """

        fuse_modules(self, ['conv1', 'bn1', 'relu'], inplace=True)
        for m in self.modules():
            if type(m) == Bottleneck or type(m) == BasicBlock:
                m.fuse_model()
    

def resnet(cfg, sketch_rate=None, start_conv=1, num_classes=1000):
    if cfg == 'resnet18':
        return ResNet(BasicBlock, [2, 2, 2, 2], sketch_rate=sketch_rate, num_classes=num_classes, start_conv=start_conv)
    elif cfg == 'resnet34':
        return ResNet(BasicBlock, [3, 4, 6, 3], sketch_rate=sketch_rate, num_classes=num_classes, start_conv=start_conv)
    elif cfg == 'resnet50':
        return ResNet(Bottleneck, [3, 4, 6, 3], sketch_rate=sketch_rate, num_classes=num_classes, start_conv=start_conv)
    elif cfg == 'resnet101':
        return ResNet(Bottleneck, [3, 4, 23, 3], sketch_rate=sketch_rate, num_classes=num_classes, start_conv=start_conv)
    elif cfg == 'resnet152':
        return ResNet(Bottleneck, [3, 8, 36, 3], sketch_rate=sketch_rate, num_classes=num_classes, start_conv=start_conv)

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])