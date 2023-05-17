import tensorrt as trt
import os
import numpy as np
import time
from cuda import cudart
import argparse

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--trtFile', default='', type=str, help='trt engine file path')
parser.add_argument('--onnxFile', default='',type=str, help='onnx file path')
parser.add_argument('--pthFile', default='',type=str, help='pth file path')
parser.add_argument('--batch_size', default=16,type=int, help='batch size to inference')
parser.add_argument('--opt_batch', type=int, default=4, help='context optimal batch size')
parser.add_argument('--max_batch', type=int, default=16, help='context max batch size')
parser.add_argument('--Sparsity', action='store_true', help='Sparsity Enable option default=True')
parser.add_argument('--NotUseFP16Mode', action='store_false', help='Convert TF32 to FP16')
parser.add_argument('--CIFAR100', action='store_true', help='CIFAR100')
args = parser.parse_args()

NotUseFP16Mode = args.NotUseFP16Mode
Sparsity = args.Sparsity
onnxFile = args.onnxFile
trtFile = args.trtFile
nHeight, nWidth = 224, 224
if args.CIFAR100 :
    nHeight, nWidth = 32, 32
batch_size = args.batch_size
opt_batch = args.opt_batch
max_batch = args.max_batch
# Parse network, rebuild network and do inference in TensorRT ------------------
#set classes
logger = trt.Logger(trt.Logger.WARNING)  # create Logger, avaiable level: VERBOSE, INFO, WARNING, ERRROR, INTERNAL_ERROR
builder = trt.Builder(logger)  # create Builder
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))  # create Network
profile = builder.create_optimization_profile()  # create Optimization Profile if using Dynamic Shape mode
config = builder.create_builder_config()  # create BuidlerConfig to set meta data of the network

if Sparsity:
    config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)  # for Sparsity option test
    print(f'set Sparsity : {Sparsity}')

if NotUseFP16Mode:
    config.set_flag(trt.BuilderFlag.FP16)
    print('***FP16 Mode***')
else :
    config.set_flag(trt.BuilderFlag.TF32)
    print('***TF32 Mode***')

parser = trt.OnnxParser(network, logger)
if not os.path.exists(onnxFile):
    print("Failed finding ONNX file!")
    exit()
print("Succeeded finding ONNX file!")

#parsing onnx
with open(onnxFile, "rb") as model:
    if not parser.parse(model.read()):
        print("Failed parsing .onnx file!")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit()
    print("Succeeded parsing .onnx file!")

# batchsize == -1  / explicit batch 로 설정하기 때문에(추론 할 때, batch_size를 유동적으로 변경 가능- 세팅한 범위까지)
inputTensor = network.get_input(0) # shpae == [-1,3,32,32]

print(list(inputTensor.shape))

# shape을 여러개의 (batch,1,h,w)로 넣음(onnx의 dynamic_axes옵션 넣어서 batch의 자리를 지정해야함)
profile.set_shape(inputTensor.name, [1, 3, nHeight, nWidth], [opt_batch, 3, nHeight, nWidth], [max_batch, 3, nHeight, nWidth])#여기에 batch 설정
print("OptimizationProfile is available? %s" % profile.__nonzero__())
config.add_optimization_profile(profile)

engineString = builder.build_serialized_network(network, config)  # create a serialized network

if engineString == None:
    print("Failed building engine!")
    exit()
print(f"Succeeded building engine!, Sparsity : {Sparsity}, opt_batch : {opt_batch}, max_batch : {max_batch}")

with open(trtFile, "wb") as f:  # create engine
    f.write(engineString)

#### if forawrd return has another valuse except 'x' we have to set this ones
# identityLayer = network.add_identity(inputTensor)
# network.mark_output(identityLayer.get_output(0))
# network.unmark_output(network.get_output(0))  # remove output tensor "y"(Net output이 y,z인데 y값(logit) 제거)
