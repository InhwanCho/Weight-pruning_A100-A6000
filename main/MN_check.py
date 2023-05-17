import torch
import torchvision
import argparse
parser = argparse.ArgumentParser(description='M:N ratio check, pruning ratio check using pth file')
parser.add_argument('--CIFAR100', action='store_true', help='CIFAR100')
parser.add_argument('--pthFile', default='',type=str, help='pth file path')
args = parser.parse_args()
# ############ todo M:N 확인하는 코드 #############

# PATH = './output/cifar_resnet56_best_pruned.pth'
PATH = args.pthFile
import_model = torchvision.models.resnet50(weights=None)
if args.CIFAR100 :
    import_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet56", pretrained=False)
load = torch.load(PATH)

print('dick keys : ',dict(load).keys())
print('best_acc : ',load['best_acc1'])
# print(dict(load['state_dict']).keys())
total = 0.
nonzero = 0.
a = torch.Tensor(load['state_dict']['layer1.0.conv1.__weight_mma_mask'])
if args.CIFAR100 :
    a = torch.Tensor(load['layer1.0.conv1.__weight_mma_mask'])
print('number of non_zero parameters : ',sum(a.view(-1)))
print('number of trainable parameters : ',a.numel())
print('pruning ratio : ',1- sum(a.view(-1)) / a.numel())

# print(load['layer3.0.conv1.__weight_mma_mask'].shape)#[16, 16, 3, 3]

mask = load['state_dict']['layer1.0.conv1.__weight_mma_mask']
if args.CIFAR100 :
    mask = load['layer1.0.conv1.__weight_mma_mask']
mask = mask.cpu().numpy().astype(int)
mask = torch.Tensor(mask)
shape = mask.shape

M ,N= 4, 2
MN_check = mask.permute(2,3,0,1).contiguous().view(shape[2]*shape[3]*shape[0], shape[1])
# MN_check = MN_check.reshape(int(a.numel()/M),M)
print('mask shape : ',shape)


#[1,1,0,0] ... [1,0,1,0] 이런 식으로 4:2면 4개 당 2개 이상의 0이 존재해야함
print('just show two masks in detail below')
print(MN_check[0])
print(MN_check[1])

total_num = []
print('reshaped MN shape : ',MN_check.shape)
for i in range(MN_check.shape[0]):
    for j in range(int(MN_check.shape[1]/M)):
        if sum(MN_check[i][j*M:(j*M)+M]) == N:
            total_num.append(1)
print('number of trainable parameters (after pruning) : ',sum(total_num))
print('number of trainable parameters / M (must be the same as above): ',(int(a.numel()/M)))
if int(a.numel()/M) == sum(total_num) :
    print(f'ratio == {M}:{N}, it pruned well')
else :
    print('check pruning code')