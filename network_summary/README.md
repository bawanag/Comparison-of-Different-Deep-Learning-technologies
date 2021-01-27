# Model basic information static

## Introduction 
summary the basic information of models, include parameter number, model evaluation size, and GFLOPs


## "torchsumarry_detect" usage

- `python torchsummary_detect.py" -m resnet18
- `python torchsummary_detect.py" -m vgg11

## MobileNetv2_summary, MobileNetv1_summary, LeNet5 usage
because the models above cannot be implemented by Pytorch hub, the model structures are written in code and detect the basic information separately.
- `python MobileNetv1_summary.py
- `python torchsummary_detect.py
- `python torchsummary_detect.py


