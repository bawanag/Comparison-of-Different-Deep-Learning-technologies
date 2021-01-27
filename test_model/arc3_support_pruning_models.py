# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Test Model performance
# %% [markdown]
# ## Identity Model
 

# %%
import torch
import os
import time
import datetime
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import json
import argparse
import torch
import logging
import torchvision.models as models
import xlutils
import numpy as np
import os
import psutil
from Database_util import TestDatabaseUtil
import urllib
import pynvml
import filter_sketch_pruning.common as utils
import filter_sketch_pruning.resnet_imagenet as resnet_pruning
import system_information_unit as cpu_info

# load the memory monitor
process = psutil.Process(os.getpid())

logging.basicConfig(level=logging.INFO,
                    filename='../log/test_arc3.log',
                    filemode='a',
                    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )
console_handler = logging.StreamHandler()

logger = logging.getLogger(__name__)
logger.addHandler(console_handler)
logger.info('########################################################################' +
            str(datetime.datetime.now()))

default_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--device', default=default_device, type=str,
                    help='test device cuda or cpu(default: if cuda available, using cuda)')
parser.add_argument('-m', '--model', default='resnet/resnet18.pth', type=str,
                    help='path of test model (default: resnet/resnet18.pth)')
parser.add_argument('-q', '--quantization',
                    action='store_true',  help='open quantization model test')
parser.add_argument('-p', '--pruning',
                    action='store_true',  help='open pruning model test')
parser.add_argument('-l', '--local',
                    action='store_true',  help='is it local computer running or not')
parser.add_argument('-s', '--state_dict', default="", type=str,
                    help='whether load the state_dict model, if load it, appedix the model type eg. mobilenet_v2, resnet18, vgg11_bn(default: if cuda available, using cuda)')
parser.add_argument('-sr', '--sketch_rate', default="", type=str,
                    help='the rate of pruning')
args = parser.parse_args()

if True == args.local:
    root_location = '../../'
else:
    root_location = '/nobackup/sc19yt/project/pytorch_model_zoo_pretrain_model/'
root_imageset_dir = root_location + \
    "my_imagenet/ImageNet_validset/"  # identify the classification, imagenet1k validaset standard
class_label_location = root_location + \
    "my_imagenet/scrip/imagenet_class_index.txt"


# %%
def get_cuda_memory_usage():
    if args.device == "cuda":
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        cudamen_comsumption = gpu_meminfo.used/1044**2  # unit MB
        return cudamen_comsumption
    else:
        return 0


# %%
before_load_models_hostmen_comsumption = process.memory_info().rss
before_load_models_cudamen_comsumption = get_cuda_memory_usage()
logger.info(args)
device = torch.device(args.device)

if args.model == '../quantization_models/inception_v3.pth':
    model = models.quantization.inception_v3(pretrained=True,quantize=True).to(device)
elif args.model == '../shufflenet/quantization/shufflenet_v2_x1_0.pth':
    device = 'cpu'
    model = models.quantization.shufflenet_v2_x1_0(pretrained=True,quantize=True).to(device)
else:    
    if True == args.quantization:
        model = torch.jit.load(root_location + args.model).to(device)
    elif True == args.pruning:
        model = resnet_pruning.resnet('resnet50', sketch_rate=utils.get_sketch_rate(args.sketch_rate), start_conv=1).to(device)
        ckpt = torch.load(root_location + args.model, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
    else:
        if args.state_dict == "":
            model = torch.load(root_location + args.model).to(device)
        else:
            model = models.__dict__[args.state_dict]()
            model.load_state_dict(torch.load(root_location + args.model))
            model.to(device)


# device = 'cuda'
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model = torch.load(root_location + 'xception.pth').to(device)
# model = torch.load(root_location + 'densenet201.pth').to(device)
# model = torch.load(root_location + 'vgg11_pruning_fc.pth').to(device)

# model = torch.load(root_location + 'vgg/vgg11.pth').to(device)
# model = torch.load(root_location + 'vgg/vgg11_bn.pth').to(device)
# model = torch.load(root_location + 'vgg/vgg13.pth').to(device)
# model = torch.load(root_location + 'vgg/vgg13_bn.pth').to(device)
# model = torch.load(root_location + 'vgg/vgg16.pth').to(device)
# model = torch.load(root_location + 'vgg/vgg16_bn.pth').to(device)
# model = torch.load(root_location + 'vgg/vgg19.pth').to(device)
# model = torch.load(root_location + 'vgg/vgg19_bn.pth').to(device)

# model = torch.load(root_location + 'resnet/resnet18_pruning.pth').to(device)
# model = torch.load(root_location + 'resnet/resnet18.pth').to(device)
# model = torch.load(root_location + 'resnet/resnet34.pth').to(device)
# model = torch.load(root_location + 'resnet/resnet50.pth').to(device)
# model = torch.load(root_location + 'resnet/resnet101.pth').to(device)
# model = torch.load(root_location + 'resnet/resnet152.pth').to(device)
# model = torch.load(root_location + 'inception_v3/inception_v3.pth').to(device)
# model = torch.load(root_location + 'shufflenet/shufflenet_v2_x0_5.pth').to(device)
# model = torch.load(root_location + 'shufflenet/shufflenet_v2_x1_0.pth').to(device)


# model = torch.jit.load(root_location + 'resnet/quantization/resnet18.pth').to(device)
# model = torch.jit.load(root_location + 'resnet/quantization/resnext101_32x8d.pth').to(device)
# model = torch.jit.load(root_location + 'quantization_models/mobilenet_v2.pth').to(device)
# model = torch.jit.load(root_location + 'quantization_models/inception_v3.pth').to(device)
# model = torch.load(root_location + 'shufflenet/quantization/shufflenet_v2_x1_0.pth').to(device)



criterion = nn.CrossEntropyLoss()
logger.info(device)
model.eval()
# reporter = MemReporter()
# reporter.report()


# inp = torch.Tensor(1,3,224,224).to(device)
# pass in a model to automatically infer the tensor names
# reporter = MemReporter(model)
# out = model(inp).mean()
after_load_models_hostmen_comsumption = process.memory_info().rss
host_memory_usage = (after_load_models_hostmen_comsumption -
                     before_load_models_hostmen_comsumption) / 1024 / 1024  # unit MB

after_load_models_cudamen_comsumption = get_cuda_memory_usage()
cuda_memory_usage = after_load_models_cudamen_comsumption - \
    before_load_models_cudamen_comsumption
print('Used Host Memory:', host_memory_usage, 'MB')
print('Used CUDA Memory:', cuda_memory_usage, 'MB')

if args.device == "cpu":
    ctdu = TestDatabaseUtil('../log/cpu_test_database.xls')
else:
    ctdu = TestDatabaseUtil('../log/gpu_test_database.xls')
ctdu.add_execute_current_time(datetime.datetime.now())
ctdu.add_model_path(args.model)
ctdu.add_cpu_info(cpu_info.get_cpu_info())
ctdu.add_model_host_memory_usage(host_memory_usage)
ctdu.add_model_cuda_memory_usage(cuda_memory_usage)


# %%
class_names = []
with open(class_label_location) as f:
    lines = f.readlines()
    class_to_lable_json = lines[0]
    class_to_lable_dict = json.loads(s=class_to_lable_json)
    for key in class_to_lable_dict:
        class_names.append(class_to_lable_dict[key][0])


def get_class_from_path(value):
    for key in class_to_lable_dict:
        if value == class_to_lable_dict[key][0]:
            return int(key)


# %%
def get_meta(root_dir, dirs):

    paths, classes = [], []
    for i, dir_ in enumerate(dirs):
        for entry in os.scandir(root_dir + dir_):
            if (entry.is_file()):
                paths.append(entry.path)
                classes.append(get_class_from_path(dir_))

    return paths, classes


# %%
# Benign images we will assign class 0, and malignant as 1
paths, classes = get_meta(root_imageset_dir, class_names)

data = {
    'path': paths,
    'class': classes
}

data_df = pd.DataFrame(data, columns=['path', 'class'])
data_df = data_df.sample(frac=1).reset_index(drop=True)  # Shuffles the data


# %%
batch_size = 200  # 50000%batch_size = 0
# batch_size = 1024
log_time = 10
validset_size = len(data_df)
log_interval = int(validset_size/(batch_size*log_time))
print(len(data_df))
print(data_df.head())


# %%
class ImageNet1000(Dataset):

    def __init__(self, df, transform=None):

        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load image from path and get label
        x = Image.open(self.df['path'][index])
        try:
            # To deal with some grayscale images in the data
            x = x.convert('RGB')
        except:
            pass
        y = torch.tensor(int(self.df['class'][index]))

        if self.transform:
            x = self.transform(x)

        return x, y


# %%
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

ins_dataset_valid = ImageNet1000(
    df=data_df,
    transform=data_transform,
)
valid_dataset_loader = torch.utils.data.DataLoader(
    ins_dataset_valid,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)


# %%
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# %%
def single_valid_dataset():
    print('=====================================================================')
    print("\033[1;33;44m"+"Test model architecture: "+ args.model + "\033[0m'")
    running_loss = 0.0
    correct = 0
    total = 0
    acc1 = []
    acc5 = []
    outputs_list = []

    starttime = 0
    nb_classes = 1000
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for i, data in enumerate(valid_dataset_loader, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            if i == 0:
                starttime = starttime_each_log = datetime.datetime.now()
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            acc1_temp, acc5_temp = accuracy(outputs, labels, topk=(1, 5))
            acc1.append(acc1_temp)
            acc5.append(acc5_temp)
            if i % log_interval == 0 and i != 0:
                endtime_each_log = datetime.datetime.now()
                time_comsumption = endtime_each_log - starttime_each_log
                starttime_each_log = endtime_each_log
                print('validset    Loss: {:.4f}   Top1 Accuracy : {:.4f}%  Top5 accuracy : {:.4f}%  Time comsumption:{}'.format(
                    running_loss/(i*batch_size), acc1_temp.item(), acc5_temp.item(), time_comsumption))
                cuda_memusage = get_cuda_memory_usage()
                print('Used CUDA Memory:', cuda_memusage, 'MB')
                host_memory_usage = process.memory_info().rss/1024**2
                print('Used Host Memory:', host_memory_usage, 'MB')
            _, preds = torch.max(outputs, 1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    print("===========================================================================")
    print(confusion_matrix)  # TODO too much data and not print
    precision = confusion_matrix.diag()/confusion_matrix.sum(0)
    ctdu.add_precision(str(precision))
    logger.info('precision:'+str(precision))
    recall = confusion_matrix.diag()/confusion_matrix.sum(1)
    ctdu.add_recall(str(recall))
    logger.info('recall:'+str(recall))
    f1 = 2*precision*recall/(precision+recall)
    ctdu.add_F1_score(str(f1))
    logger.info('recall:'+str(f1))

    acc1_sum = 0
    for acc1_temp in acc1:
        acc1_sum += acc1_temp
    acc1_average = acc1_sum/len(acc1)
    acc5_sum = 0
    for acc5_temp in acc5:
        acc5_sum += acc5_temp
    acc5_average = acc5_sum/len(acc5)

    dataset_loss = running_loss/(i*batch_size)
    overall_time_comsumption = datetime.datetime.now() - starttime
    ctdu.add_running_time(str(overall_time_comsumption))
    ctdu.add_cross_entropy_loss(dataset_loss)
    ctdu.add_Top1_Accuracy(acc1_average.item())
    ctdu.add_Top5_Accuracy(acc5_average.item())
    logger.info('overall validset    Loss: {:.4f}   Top1 Accuracy : {:.4f}%  Top5 accuracy : {:.4f}% Time comsumption:{}'.format(
        dataset_loss, acc1_average.item(), acc5_average.item(), overall_time_comsumption))


# %%
url, filename = (
    "https://github.com/pytorch/hub/raw/master/dog.jpg", "dog.jpg")
try:
    urllib.URLopener().retrieve(url, filename)
except:
    urllib.request.urlretrieve(url, filename)

input_image = Image.open(filename)
input_tensor = data_transform(input_image)
input_batch = input_tensor.unsqueeze(0).to(device)
start = datetime.datetime.now()
with torch.no_grad():
    result = model(input_batch)
result.cpu()
end = datetime.datetime.now()
inference_time = end-start
ctdu.add_Inference_time(str(inference_time))
logger.info('inference time:'+str(inference_time))

# %%
single_valid_dataset()
ctdu.save_worksheet()
logger.info(
    '########################################################################\n\n')
