import torch.backends.cudnn as cudnn
import torchvision
import torch
from pprint import pprint
from torch.autograd import Variable
from data.dataloader_cifar100 import get_data_loader
import time
from apex.contrib.sparsity import ASP
import numpy as np
import os

np.random.seed(111)
torch.manual_seed(111)
torch.cuda.manual_seed_all(111)
cudnn.deterministic = True
cudnn.benchmark = False
model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet56", pretrained=True).cuda()
# load = torch.load('./output/cifar_resnet56_best_pre.pth')
# load = torch.load('./cifar_resnet56_best_pruned.pth')
# model.load_state_dict(load,strict=False)
# model = torch.nn.DataParallel(model)

data_Path = os.path.dirname(os.path.realpath(__file__)) + '/data/'
nTrainBatchSize = 256
epochs = 2
train_set, test_set = get_data_loader(data_Path, nTrainBatchSize)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=nTrainBatchSize, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=nTrainBatchSize, shuffle=False)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.1,momentum=0.9, weight_decay=0.0005,nesterov=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(T_max=200, eta_min=0,optimizer=optimizer)
#
MNmode = True
# onnxFile = './resnet56_pruned.onnx'
# nHeight, nWidth = 32, 32


def train(model, train_loader):
    model.train()
    train_acc, correct_train, train_loss, target_count = 0, 0, 0, 0
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda()
        input_var = Variable(input)
        target_var = Variable(target)

        optimizer.zero_grad()
        output = model(input_var)
        loss = criterion(output, target_var)
        loss.backward()
        optimizer.step()

        # accuracy
        _, predicted = torch.max(output.data, 1)
        target_count += target_var.size(0)
        correct_train += (target_var == predicted).sum().item()
        train_acc = (100 * correct_train) / target_count
        # scheduler.step()

    return train_acc
def validate(model, test_loader,print_=None):
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
            _, predicted = torch.max(output.data, 1)
            target_count += target_var.size(0)
            correct_val += (target_var == predicted).sum().item()
            val_acc = 100 * correct_val / target_count
            ######## val_loss = (val_acc * 100) / target_count, val_loss / target_count

        if print_ :
            print(f'total val_acc : {val_acc}')

        return val_acc
# validate(model,test_loader,print_=True) # 72.63 # (71.14,71.05)
# ###
# # pre_train_acc = []
# # pre_val_acc = [0]
# # best_val_acc = [0]
# # for epoch in range(epochs):
# #     train_acc = train(model, train_loader)
# #     val_acc = validate(model, test_loader)
# #     print("Epoch {0}: train_acc {1} \t val_acc {2} \t".format(epoch, train_acc, val_acc))
# #     pre_train_acc.append(train_acc)
# #     pre_val_acc.append(val_acc)
#     # best_val_acc.append(best_val_acc)
# #
# #
# # print(pre_train_acc)
# # print(pre_val_acc)
# # print(best_val_acc)
# # ###
# if MNmode:
#     validate(model, test_loader, print_=True)
#     #####todo 1. After pre-trained load the model
#     PATH = './cifar_resnet56_best_pruned.pth'  # pth
#
#     #####todo 2. M:N pruning to reduce the time to calculate
#     ASP_time = time.perf_counter()
#     ASP.prune_trained_model(model, optimizer)
#     print(f'ASA pruning time : {time.perf_counter() - ASP_time}')
#     print('complete the ASA prune **********')
#     torch.save(model.state_dict(), PATH)
#     # model.load_state_dict(torch.load(PATH),strict=True)
#     print('pth file has been saved')
#     ######### fine-tuning
#     validate(model, test_loader, print_=True)
#     epochs_ = epochs
#     fine_train_acc = []
#     fine_val_acc = []
#     pre_train_acc = []
#     pre_val_acc = [0]
#     best_val_acc = [0]
#     for epoch in range(epochs_):
#         train_acc = train(model, train_loader)
#         val_acc = validate(model, test_loader)
#         print("Epoch {0}: train_acc {1} \t val_acc {2} \t".format(epoch, train_acc, val_acc))
#         fine_train_acc.append(train_acc)
#         fine_val_acc.append(val_acc)
#         best_val_acc.append(best_val_acc)
#
#     # #####todo 3. After pre-trained load the model
#
#     torch.save(model.state_dict(), PATH)
#     print(PATH)
#     model.load_state_dict(torch.load(PATH),strict=True)
#     model.eval()
#     validate(model, test_loader,print_=True)
#     print(f'pre val acc :{pre_val_acc}, fine train acc :{fine_train_acc}')
#     print(f'pre val acc :{pre_val_acc}, fine train acc :{max(fine_train_acc)}')
#     print(f'pre val acc :{pre_val_acc}, fine val acc :{fine_val_acc}')
#     print(f'pre val acc :{pre_val_acc}, fine val acc :{max(fine_val_acc)}')
#
# # #
# # torch.onnx.export(model, torch.randn(32,3,nHeight,nWidth,device='cuda'),onnxFile, input_names=["x"], output_names=["z"], verbose=False, keep_initializers_as_inputs=True, do_constant_folding=True,opset_version=12, dynamic_axes={"x": {0: "nBatchSize"}, "z": {0: "nBatchSize"}})
# # print('converted onnx')



# #########todo train imageNet
# import torchvision.transforms as transforms
# batch_size = 256
# workers = 4
# lr = 0.1
# momentum = 0.9
# weight_decay = 1e-4
# train_sampler = None
# val_sampler = None
# traindir = '/home/keti/workspace/Dataset/imagenet/train'
# valdir = '/home/keti/workspace/Dataset/imagenet/val'
# MNmode = True
# epochs = 1
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
# train_dataset = torchvision.datasets.ImageFolder(
#     traindir,
#     transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         normalize,
#     ]))
#
# val_dataset = torchvision.datasets.ImageFolder(
#     valdir,
#     transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         normalize,
#     ]))
#
# train_loader = torch.utils.data.DataLoader(
#     train_dataset, batch_size=batch_size, shuffle=True,
#     num_workers=workers, pin_memory=True, sampler=train_sampler)
#
# test_loader = torch.utils.data.DataLoader(
#     val_dataset, batch_size=batch_size, shuffle=False,
#     num_workers=workers, pin_memory=True, sampler=val_sampler)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# criterion = torch.nn.CrossEntropyLoss().to(device)
# model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights).cuda()
# model = torch.nn.DataParallel(model).cuda()
# model = torch.nn.DataParallel(model).to(device)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr,
#                             momentum=momentum,
#                             weight_decay=weight_decay)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
#
# validate(model=model,test_loader=test_loader,print_=True)

if MNmode:
    # validate(model, test_loader, print_=True)
    #####todo 1. After pre-trained load the model

    #####todo 2. M:N pruning to reduce the time to calculate
    ASP_time = time.perf_counter()
    ASP.prune_trained_model(model, optimizer)
    print(f'ASA pruning time : {time.perf_counter() - ASP_time}')
    print('complete the ASA prune **********')
    ######### fine-tuning
    validate(model, test_loader, print_=True)
    epochs_ = epochs
    fine_train_acc = []
    fine_val_acc = []
    pre_train_acc = []
    pre_val_acc = [0]
    best_val_acc = [0]
    for epoch in range(epochs_):
        train_acc = train(model, train_loader)
        val_acc = validate(model, test_loader)
        print("Epoch {0}: train_acc {1} \t val_acc {2} \t".format(epoch, train_acc, val_acc))
        fine_train_acc.append(train_acc)
        fine_val_acc.append(val_acc)
        best_val_acc.append(best_val_acc)

    # #####todo 3. After pre-trained load the model
    PATH = './tem_folder/CIFAR100_res56.pth'  # pth

    torch.save(model.state_dict(), PATH)
    print(PATH)
    model.load_state_dict(torch.load(PATH),strict=True)
    model.eval()
    validate(model, test_loader,print_=True)
    print(f'pre val acc :{pre_val_acc}, fine train acc :{fine_train_acc}')
    print(f'pre val acc :{pre_val_acc}, fine train acc :{max(fine_train_acc)}')
    print(f'pre val acc :{pre_val_acc}, fine val acc :{fine_val_acc}')
    print(f'pre val acc :{pre_val_acc}, fine val acc :{max(fine_val_acc)}')

# onnxFile = './output/imagenet_res50.onnx'
onnxFile = './tem_folder/CIFAR100_res56.onnx'
print('****************')
# print(model.module)
# print(model.state_dict().keys())
dummy = torch.randn(8,3,32,32,device='cuda')
torch.onnx.export(model,dummy ,onnxFile, input_names=["x"], output_names=["z"], verbose=False, keep_initializers_as_inputs=True, do_constant_folding=True,opset_version=12, dynamic_axes={"x": {0: "nBatchSize"}, "z": {0: "nBatchSize"}})
print('converted onnx')

# odict_keys(['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean' (기존)
# odict_keys(['module.conv1.weight', 'module.bn1.weight', 'module.bn1.bias', 'module.bn1.running_mean',(DP)
