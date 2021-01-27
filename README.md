# Compare and analysis different Deep Learning models and compression technologies toward choose efficient Convolution Neutral Network architecture

## Introduction 
In this project, it already download all the model I will test it, all the model also could be convert ot quantized version, and test each different architecture model. Also, the models which are pre-process， test and compared with each other.

If want to see the detail of each method, please enter the path to see sub-README.MD


## path structure 
log ------- the test log about each pretrained and pre-process models

network_summary ------- static all the network parameter number and some charactristic

pretrain_model_download ------ the code which can download the pretrained model on Pytorch zoo

pruning ----- pruning pretrained models which are download on pytorch zoo. include channel and weight pruning 

quantization ----- generate quantization models 

statistics ------- statistics the log information and the data which can be collected by myself

test_model -------- the test model scrips

test_queue ------- test model HPC jobs queues shell auto scrips 

util ---------- some tool 

