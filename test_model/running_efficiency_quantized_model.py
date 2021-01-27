import datetime
start = datetime.datetime.now()
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import argparse
from torchvision import models
import os
from Database_util import TestDatabaseUtil
import psutil
import filter_sketch_pruning.common as utils
import filter_sketch_pruning.resnet_imagenet as resnet_pruning
import urllib
import system_information_unit as cpu_info
process = psutil.Process(os.getpid())
before_load_models_hostmen_comsumption = process.memory_info().rss

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default='resnet/resnet18.pth', type=str,
                    help='path of test model (default: resnet/resnet18.pth)')
parser.add_argument('-q', '--quantization',
                    action='store_true',  help='disables CUDA training')
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
    img_root_location = '../../my_imagenet/ImageNet_test_model_efficiency'
else:
    root_location = '/nobackup/sc19yt/project/pytorch_model_zoo_pretrain_model/'
    img_root_location = '/nobackup/sc19yt/project/pytorch_model_zoo_pretrain_model/my_imagenet/ImageNet_test_model_efficiency'

device = torch.device('cpu')

if args.model == '../quantization_models/inception_v3.pth':
    model = models.quantization.inception_v3(pretrained=True,quantize=True).to(device)
elif args.model == '../shufflenet/quantization/shufflenet_v2_x1_0.pth':
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

after_load_models_hostmen_comsumption = process.memory_info().rss
host_memory_usage = (after_load_models_hostmen_comsumption -
                     before_load_models_hostmen_comsumption) / 1024 / 1024  # unit MB
ctdu = TestDatabaseUtil('../log/test_model_running_effciency_cpu_quantized_model.xls')
ctdu.add_execute_current_time(datetime.datetime.now())
ctdu.add_model_path(args.model)
ctdu.add_model_host_memory_usage(host_memory_usage)
ctdu.add_cpu_info(cpu_info.get_cpu_info())
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

### Test inference time
url, filename = (
    "https://github.com/pytorch/hub/raw/master/dog.jpg", "dog.jpg")
try:
    urllib.URLopener().retrieve(url, filename)
except:
    urllib.request.urlretrieve(url, filename)

input_image = Image.open(filename)
input_tensor = data_transform(input_image)
input_batch = input_tensor.unsqueeze(0).to(device)
with torch.no_grad():
    result = model(input_batch)
result.cpu()
end = datetime.datetime.now()
inference_time = end-start
ctdu.add_Inference_time(str(inference_time))
print('model path:'+args.model +'   inference time:'+str(inference_time))

def get_meta(img_root_dir):
    paths = []
    for entry in os.scandir(img_root_dir):
        if (entry.is_file()):
            paths.append(entry.path)
    return paths

class ImageNet1000picture(Dataset):

    def __init__(self, paths, transform=None):

        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # Load image from path and get label
        x = Image.open(self.paths[index])
        try:
            # To deal with some grayscale images in the data
            x = x.convert('RGB')
        except:
            pass

        if self.transform:
            x = self.transform(x)

        return x

img_paths = get_meta(img_root_location)

ins_dataset_valid = ImageNet1000picture(
    paths=img_paths,
    transform=data_transform,
)
test_efficiency_dataset_loader = torch.utils.data.DataLoader(
    ins_dataset_valid,
    batch_size=128,
    shuffle=False,
    num_workers=2
)



### Test running accuracy
start = datetime.datetime.now()
with torch.no_grad():
        for i, data in enumerate(test_efficiency_dataset_loader, 0):
            images = data
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

end = datetime.datetime.now()
overall_time_comsumption = end-start
ctdu.add_running_time(overall_time_comsumption)
ctdu.save_worksheet()
print('model path:'+args.model +'   overall time comsumption:'+str(overall_time_comsumption))
