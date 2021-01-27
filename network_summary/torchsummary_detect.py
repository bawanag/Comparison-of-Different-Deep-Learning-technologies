import torch
import torch.nn as nn
from torchsummary import summary
from torchvision import models
from pthflops import count_ops
import argparse
import filter_sketch_pruning.resnet_imagenet as resnet_pruning
import filter_sketch_pruning.common as utils

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_name', default='mobilenet_v2', type=str,
                    help='model name(default: mobilenet_v2)')
parser.add_argument('-p', '--pruning',
                    action='store_true',  help='open pruning model test')
parser.add_argument('-mp', '--model_path', default='sketch_resnet50_0.2.pt', type=str,
                    help='path of test model (default: resnet/resnet18.pth)')
parser.add_argument('-sr', '--sketch_rate', default="[0.2]*16python torchsummary_detect.py -p -mp sketch_resnet50_0.6 -sr [0.6]*16", type=str,
                    help='the rate of pruning')
args = parser.parse_args()
print(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_location_pruning_models = '../../pruning_models/'
if True == args.pruning:
        net = resnet_pruning.resnet('resnet50', sketch_rate=utils.get_sketch_rate(args.sketch_rate), start_conv=1).to(device)
        ckpt = torch.load(root_location_pruning_models + args.model_path, map_location=device)
        net.load_state_dict(ckpt['state_dict'])
else:
    net = models.__dict__[args.model_name]().to(device)
summary(net,(3,224,224))
inp = torch.rand(1,3,224,224).to(device)
count_ops(net, inp)