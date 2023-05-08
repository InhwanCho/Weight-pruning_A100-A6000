import torch
from model import resnet
import os
import tensorrt as trt
import numpy as np
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
######### inference test ########

trtFile = args.trtFile
SPARSITY = args.Sparsity
nHeight, nWidth = 224, 224
if args.CIFAR100 :
    nHeight, nWidth = 32, 32
batch_size = args.batch_size
nWarmUp = 20
nTest = 200

f = open(trtFile, "rb")
logger = trt.Logger(trt.Logger.WARNING)
engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())  # create inference Engine using Runtime
nIO = engine.num_io_tensors  # number of input+output = 여기선 2
lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]  # nIO tensorName을 부여(onnx만들때) 여기선 [x(in),z(out-argmax)]
nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(
    trt.TensorIOMode.INPUT)  # number of input ==> 1

context = engine.create_execution_context()  # 추론하기 위한 context 클래스 객체 생성
context.set_input_shape(lTensorName[0], [batch_size, 3, nHeight, nWidth])  # 여기선 input='x'에 대한 shape
print(context.get_tensor_shape(lTensorName[0]))
for i in range(nIO):
    print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]),
          engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

### imagenet
import torchvision
import torchvision.transforms as transforms

######## ImageNet
valdir = '/home/keti/workspace/Dataset/imagenet/ILSVRC2012_img_val'
workers = 4
val_sampler = None
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
val_dataset = torchvision.datasets.ImageFolder(
    valdir,
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
test_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False,
    num_workers=workers, pin_memory=True, sampler=val_sampler)

if args.CIFAR100:
    from data.dataloader_cifar100 import get_data_loader
    data_Path = os.path.dirname(os.path.realpath(__file__)) + '/data/'
    _, test_set = get_data_loader(data_Path, batch_size)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)



dataiter = iter(test_loader)
images, labels = next(dataiter)
data = images.detach().numpy()

def run():
    bufferH = []  # prepare the memory buffer on host
    bufferH.append(np.ascontiguousarray(data))

    for i in range(nInput, nIO):
        bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]),
                                dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
    bufferD = []  # prepare the memory buffer on device
    for i in range(nIO):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):  # copy input data from host buffer into device buffer
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes,
                          cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    for i in range(nIO):  # set address of all input and output data in device buffer
        context.set_tensor_address(lTensorName[i], int(bufferD[i]))

    # for i in range(nIO):  # just to see in detail
    #     print(lTensorName[i])  # lTensorName ==> x == input, z == output
    #     print(bufferH[i].shape)  # bufferH     ==> [[batch,channel,h,w],[1,number_classes]]
    context.execute_async_v3(0)  # do inference computation
    for i in range(nWarmUp):
        context.execute_async_v3(0)  # warmup inference

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record()
    for i in range(nTest):
        context.execute_async_v3(0)  # inference : The cuda stream on which the inference kernels will be enqueued
    ender.record()
    torch.cuda.synchronize()
    print('\033[34m',f'sparsity : {SPARSITY}, {nTest} times test per inference : {starter.elapsed_time(ender) / nTest :.3f}')
    print(f'batch_size : {batch_size}, trt_file : {trtFile.split("/")[-1]} \033[37m')

    for i in range(nInput, nIO):  # copy output data from device buffer into host buffer
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes,
                          cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    for b in bufferD:  # free the GPU memory buffer after all work
        cudart.cudaFree(b)
    return

run()





# # ############ todo M:N 확인하는 코드 #############
# # # PATH = 'cifar_resnet110.pth'
# PATH = './output/cifar_resnet56_best_pruned.pth'
# import_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet56", pretrained=False)
# load = torch.load(PATH)
# # import_model.load_state_dict(load,strict=False)
# # print(dict(load).keys())
# # total = 0.
# # nonzero = 0.
# # a = torch.Tensor(load['layer1.0.conv1.__weight_mma_mask'])
# # print(sum(a.view(-1)))
# # print(a.numel())
# # print(1- sum(a.view(-1)) / a.numel())
#
# # print(load['layer3.0.conv1.__weight_mma_mask'].shape)#[16, 16, 3, 3]
# a = torch.Tensor(load['layer1.0.conv1.__weight_mma_mask'])
# mask = load['layer1.0.conv1.__weight_mma_mask']
# mask = mask.cpu().numpy().astype(int)
# mask = torch.Tensor(mask)
# shape = mask.shape
# print(shape)
# M ,N= 4, 2
# MN_check = mask.permute(2,3,0,1).contiguous().view(shape[2]*shape[3]*shape[0], shape[1])
# # MN_check = MN_check.reshape(int(a.numel()/M),M)
# print(MN_check.shape)
# print(sum(MN_check[0]))
# print(MN_check[0])
# print(MN_check[1])
#
# total = []
# print(MN_check.shape)
# for i in range(MN_check.shape[0]):
#     for j in range(int(MN_check.shape[1]/M)):
#         if sum(MN_check[i][j*M:(j*M)+M]) == N:
#             total.append(1)
# print(sum(total))
# print((int(a.numel()/M)))
#
# if int(a.numel()/M) == sum(total) :
#     print(f'{M}:{N}')
