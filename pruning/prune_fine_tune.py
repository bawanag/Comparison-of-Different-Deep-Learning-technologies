import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))



import torch
import argparse
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn 
import numpy as np 
import json
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

root_location = '/nobackup/sc19yt/project/pytorch_model_zoo_pretrain_model/'
root_train_imageset_dir = root_location + 'pruning_finetune_imagenet/train/'
root_valid_imageset_dir = root_location + 'pruning_finetune_imagenet/valid/'


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--total_epochs', type=int, default=20)
parser.add_argument('--step_size', type=int, default=15)
parser.add_argument('--round', type=int, default=1)
parser.add_argument('-m', '--model', default='resnet/resnet18.pth', type=str,
                    help='path of test model (default: resnet/resnet18.pth)')
parser.add_argument('-mt', '--model_type', default='vgg', type=str,
                    help='model architecture (default: vgg)')
args = parser.parse_args()

model_type = args.model_type
def get_dataloader():
    class_label_location = root_location + \
    "my_imagenet/scrip/imagenet_class_index.txt"
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

    def get_meta(root_dir, dirs):
        paths, classes = [], []
        for i, dir_ in enumerate(dirs):
            for entry in os.scandir(root_dir + dir_):
                if (entry.is_file()):
                    paths.append(entry.path)
                    classes.append(get_class_from_path(dir_))

        return paths, classes

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
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])


    paths, classes = get_meta(root_train_imageset_dir, class_names)

    data_train = {
        'path': paths,
        'class': classes
    }

    data_df = pd.DataFrame(data_train, columns=['path', 'class'])
    data_df = data_df.sample(frac=1).reset_index(drop=True)

    batch_size = args.batch_size  # 50000%batch_size = 0
    train_size = len(data_df)
    print(len(data_df))
    print(data_df.head())

    ins_dataset_train = ImageNet1000(
        df=data_df,
        transform=data_transform,
    )
    train_loader = torch.utils.data.DataLoader(
        ins_dataset_train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )


    paths, classes = get_meta(root_valid_imageset_dir, class_names)

    data_valid = {
        'path': paths,
        'class': classes
    }

    data_df = pd.DataFrame(data_valid, columns=['path', 'class'])
    data_df = data_df.sample(frac=1).reset_index(drop=True)

    batch_size = args.batch_size  
    train_size = len(data_df)

    ins_dataset_valid = ImageNet1000(
        df=data_df,
        transform=data_transform,
    )
    test_loader = torch.utils.data.DataLoader(
        ins_dataset_valid,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, test_loader

def eval(model, test_loader):
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (img, target) in enumerate(test_loader):
            img = img.to(device)
            out = model(img)
            pred = out.max(1)[1].detach().cpu().numpy()
            target = target.cpu().numpy()
            correct += (pred==target).sum()
            total += len(target)
    return correct / total

def train_model(model, train_loader, test_loader):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    # optimizer = torch.optim.Adam(model.parameters(),0.05)
    optimizer = torch.optim.Adamax(model.parameters(), lr=0.05, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, 0.1)
    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    best_acc = -1
    for epoch in range(args.total_epochs):
        model.train()
        for i, (img, target) in enumerate(train_loader):
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(img)
            # loss = F.cross_entropy(out, target)
            # print('out size:'+str(len(out)) + '  target size:' + str(len(target)))
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            if i%50==0 and args.verbose:
                print(model_type + "  Epoch %d/%d, iter %d/%d, loss=%.4f"%(epoch, args.total_epochs, i, len(train_loader), loss.item()))
        model.eval()
        acc = eval(model, test_loader)
        print(model_type + "  Epoch %d/%d, Acc=%.4f"%(epoch, args.total_epochs, acc))
        if best_acc<acc:
            torch.save( model,  os.path.dirname(args.model) + '/' + (model_type + '-round%d.pth'%(args.round)) )
            best_acc=acc
        scheduler.step()
    print(model_type + "  Best Acc=%.4f"%(best_acc))

# from cifar_resnet import ResNet18
# import cifar_resnet as resnet

# def prune_model(model):
#     model.cpu()
#     DG = tp.DependencyGraph().build_dependency( model, torch.randn(1, 3, 32, 32) )
#     def prune_conv(conv, pruned_prob):
#         weight = conv.weight.detach().cpu().numpy()
#         out_channels = weight.shape[0]
#         L1_norm = np.sum( np.abs(weight), axis=(1,2,3))
#         num_pruned = int(out_channels * pruned_prob)
#         prune_index = np.argsort(L1_norm)[:num_pruned].tolist() # remove filters with small L1-Norm
#         plan = DG.get_pruning_plan(conv, tp.prune_conv, prune_index)
#         plan.exec()
    
#     block_prune_probs = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3]
#     blk_id = 0
#     for m in model.modules():
#         if isinstance( m, resnet.BasicBlock ):
#             prune_conv( m.conv1, block_prune_probs[blk_id] )
#             prune_conv( m.conv2, block_prune_probs[blk_id] )
#             blk_id+=1
#     return model    

def main():
    train_loader, test_loader = get_dataloader()
 
    previous_ckpt = args.model
    model = torch.load( previous_ckpt )
    print(model)
    params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Number of Parameters: %.1fM"%(params/1e6))
    train_model(model, train_loader, test_loader)


if __name__=='__main__':
    main()
