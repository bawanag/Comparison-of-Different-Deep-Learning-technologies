# Test the models

## Introduction 
using different method to test the model in the University of Leeds ARC3 HPC clusters


## Usage
### arc3.py

arc3.py -h

  -d DEVICE, --device DEVICE
                        test device cuda or cpu(default: if cuda available,
                        using cuda)
						
  -m MODEL, --model MODEL
                        path of test model (default: resnet/resnet18.pth)
						
  -q, --quantization    disables CUDA training
  
  -l, --local           is it local computer running or not
  
  -s STATE_DICT, --state_dict STATE_DICT
                        whether load the state_dict model, if load it, appedix
                        the model type eg. mobilenet_v2, resnet18,
                        vgg11_bn(default: if cuda available, using cuda)
						

float32 state dictionary model test by arc3.py using CUDA 
- 	python ../test_model/arc3.py -d cuda -m vgg/vgg11.pth

float32 state dictionary model test by arc3.py using CUDA 
- 	python ../test_model/arc3.py -d cuda -m quantization_models/resnet18_pretrained_float.pth -s resnet18

quantized model test by arc3.py using CPU
-	python ../test_model/arc3.py -d cpu -m quantization_models/resnet18_quantization_scripted_quantized.pth -q

 
### arc3_support_pruning_models.py 
arc3_support_pruning_models.py not only support arc3.py function but also support pruned model test


arc3_support_pruning_models.py -h
  -h, --help            show this help message and exit

  -d DEVICE, --device DEVICE
                        test device cuda or cpu(default: if cuda available,
                        using cuda)
						
  -m MODEL, --model MODEL
                        path of test model (default: resnet/resnet18.pth)
						
  -q, --quantization    open quantization model test
  
  -p, --pruning         open pruning model test
  
  -l, --local           is it local computer running or not
  
  -s STATE_DICT, --state_dict STATE_DICT
                        whether load the state_dict model, if load it, appedix
                        the model type eg. mobilenet_v2, resnet18,
                        vgg11_bn(default: if cuda available, using cuda)
						
  -sr SKETCH_RATE, --sketch_rate SKETCH_RATE
                        the rate of pruning
						
						
float32 pruned model test by arc3_support_pruning_models.py using CUDA 
- python ../test_model/arc3_support_pruning_models.py -d cuda -m pruning_models/sketch_resnet50_0.7.pt -sr [0.7]*16 -p


### running_efficiency.py 
Test running efficiency by cpu. put 5000 ImageNet testset Images to model and count the overall running time.


running_efficiency.py -h
usage: running_efficiency.py [-h] [-m MODEL] [-q] [-p] [-l] [-s STATE_DICT]
                             [-sr SKETCH_RATE]
							 
optional arguments:

  -h, --help            show this help message and exit
  
  -m MODEL, --model MODEL
                        path of test model (default: resnet/resnet18.pth)
						
  -q, --quantization    disables CUDA training
  
  -p, --pruning         open pruning model test
  
  -l, --local           is it local computer running or not
  
  -s STATE_DICT, --state_dict STATE_DICT
                        whether load the state_dict model, if load it, appedix
                        the model type eg. mobilenet_v2, resnet18,
                        vgg11_bn(default: if cuda available, using cuda)
						
  -sr SKETCH_RATE, --sketch_rate SKETCH_RATE
                        the rate of pruning
						

float32 state dictionary model test by running_efficiency.py
- 	python ../test_model/running_efficiency.py -m vgg/vgg11.pth


float32 state dictionary model test by running_efficiency.py using CUDA 
- 	python ../test_model/running_efficiency.py -m quantization_models/resnet18_pretrained_float.pth -s resnet18

quantized model test by running_efficiency.py using CPU(quantized model only can use x86(support AVX) or ARM cpu)
-	python ../test_model/running_efficiency.py -m quantization_models/resnet18_quantization_scripted_quantized.pth -q

float32 pruned model test by running_efficiency.py using CPU(only support ResNet50 models)
- python ../test_model/running_efficiency.py -m pruning_models/sketch_resnet50_0.7.pt -sr [0.7]*16 -p

