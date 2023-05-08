import torch
from model import resnet
import os
import tensorrt as trt
import numpy as np
from cuda import cudart
from torch.autograd import Variable
from data.dataloader_cifar100 import get_data_loader
import torchvision
import argparse

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
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

batch_size = args.batch_size
nHeight, nWidth = 224, 224
if args.CIFAR100 :
    nHeight,nWidth = 32,32
trtFile = args.trtFile

pthFile = args.pthFile
import_model = torchvision.models.resnet50(weights=None).cuda()#torchvision.models.ResNet50_Weights.DEFAULT).cuda()
# load = torch.load(pthFile)
# import_model.load_state_dict(load,strict=False)
criterion = torch.nn.CrossEntropyLoss().to('cuda')



print('###accuracy testing###')

f = open(trtFile, "rb")
logger = trt.Logger(trt.Logger.WARNING)
engine = trt.Runtime(logger).deserialize_cuda_engine(f.read()) # create inference Engine using Runtime
nIO = engine.num_io_tensors # number of input+output = 여기선 2
lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]  # nIO tensorName을 부여(onnx만들때) 여기선 [x(in),z(out-argmax)]
nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)#number of input ==> 1

context = engine.create_execution_context() #추론하기 위한 context 클래스 객체 생성
context.set_input_shape(lTensorName[0], [batch_size, 3, nHeight, nWidth]) # 여기선 input='x'에 대한 shape

def run(input_var):
    input_var = input_var.cpu().detach().numpy()
    bufferH = []  # prepare the memory buffer on host
    bufferH.append(np.ascontiguousarray(input_var))
    for i in range(nInput, nIO):
        bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
    bufferD = []# prepare the memory buffer on device
    for i in range(nIO):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])


    for i in range(nInput):# copy input data from host buffer into device buffer
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    for i in range(nIO):# set address of all input and output data in device buffer
        context.set_tensor_address(lTensorName[i], int(bufferD[i]))

    context.execute_async_v3(0) # do inference computation

    for i in range(nInput, nIO): # copy output data from device buffer into host buffer
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    # print(bufferH[1])
    bufferH[1] = torch.Tensor(bufferH[1])
    pred = torch.max(bufferH[1],1)[-1]

    for b in bufferD:  # free the GPU memory buffer after all work
        cudart.cudaFree(b)
    return pred

def validate(model, test_loader):
    model.eval()
    val_acc, correct_val, val_loss, target_count = 0, 0, 0, 0
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            input = input.cuda()
            target = target.cuda()
            input_var = Variable(input)
            target_var = Variable(target)
            output = model(input_var)
            loss = criterion(output, target_var)
            val_loss += loss.item()

            # accuracy

            predicted = run(input_var).cuda()
            target_count += target_var.size(0)
            correct_val += (target_var == predicted).sum().item()
            val_acc = 100 * correct_val / target_count
            # val_loss = (val_acc * 100) / target_count, val_loss / target_count
        print(f'val_acc : {val_acc}')
        return val_acc

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


if args.CIFAR100 :
    data_Path = os.path.dirname(os.path.realpath(__file__)) + '/data/'
    train_set, test_set = get_data_loader(data_Path, batch_size)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    import_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet56", pretrained=False).cuda()

validate(import_model,test_loader)


