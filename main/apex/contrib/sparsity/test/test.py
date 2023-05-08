import collections
import time
from itertools import permutations

import numpy as np
import torch
from torch import optim
from torchvision.models import resnet50
from apex.contrib.sparsity import ASP
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(32, 64, bias=False),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(64, 64, bias=False),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(64, 10, bias=False),
        )

    # forward
    def forward(self, x):
        x = self.classifier(x)
        return x

def mlp():
    return MLP()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device("cuda:%d" % 2)
# torch.cuda.set_device(device)
print(device)
#
model = resnet50(pretrained=True).cuda()
# model = mlp().cuda()
print(model.parameters())
# for i in model.parameters():
#     print(i.shape)
#     print(i)
torch.save(model.state_dict(), './before.pth')
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
t = time.perf_counter()

ASP.prune_trained_model(model, optimizer)
print(time.perf_counter() - t)
torch.save(model.state_dict(), './after.pth')
print(model.state_dict()['classifier.0.__weight_mma_mask'].shape)
for i in model.parameters():
    print(i.shape)
    print(i)


valid_m4n2_1d_patterns = None


def mn_2d_greedy(matrix, m, n):
    # Convert to numpy
    mat = matrix.cpu().detach().numpy()
    mask = np.ones(mat.shape, dtype=int)

    rowCount = int(mat.shape[0] / m) * m
    colCount = int(mat.shape[1] / m) * m
    for rowStartIdx in range(0, rowCount, m):
        rowEndIdx = rowStartIdx + m
        for colStartIdx in range(0, colCount, m):
            colEndIdx = colStartIdx + m
            matrixSub = np.absolute(np.squeeze(mat[rowStartIdx:rowEndIdx, colStartIdx:colEndIdx]))
            maskSub = np.squeeze(mask[rowStartIdx:rowEndIdx, colStartIdx:colEndIdx])
            maskSub.fill(0.0)
            matrixVecView = matrixSub.reshape(-1)
            maskVecView = maskSub.reshape(-1)
            linearIdx = np.argsort(matrixVecView)
            matrixIdx = [(int(x / m), x % m) for x in linearIdx]
            rowCounter = collections.Counter()
            colCounter = collections.Counter()
            for currIdx in range(len(linearIdx) - 1, -1, -1):
                currMatrixEntry = matrixIdx[currIdx]
                if (rowCounter[currMatrixEntry[0]] == n) or (colCounter[currMatrixEntry[1]] == n):
                    continue
                # end if
                maskSub[currMatrixEntry[0], currMatrixEntry[1]] = 1.0
                rowCounter[currMatrixEntry[0]] += 1
                colCounter[currMatrixEntry[1]] += 1

    return torch.tensor(mask)


def mySparsity(input):
    N, C, H, W = input.shape
    mask = torch.zeros(input.shape).cuda()
    count = int(C / 4) * 4
    for n in range(N):
        for h in range(H):
            for w in range(W):
                for beginIdx in range(0, count, 4):
                    EndIdx = beginIdx + 4
                    _, a = torch.topk(input[n, beginIdx:EndIdx, h, w], 2)
                    a[0] += beginIdx
                    a[1] += beginIdx
                    mask[n, a, h, w] = 1
    return mask * input


m = 4
n = 2
matrix = torch.rand([2, 20, 4, 4]).cuda()
# print(matrix)
# matrix2 = matrix * mn_2d_greedy(matrix, m, n)
# print(matrix2)
# print('*' * 30)
matrix2 = mySparsity(matrix)
print(matrix2)
print('*' * 30)
x = torch.arange(1, 6)
a = torch.topk(x[1:], 3)
print(a[1])

# ##########

# BATCH_SIZE=32
# dummy_input=torch.randn(BATCH_SIZE, 3, 32, 32)
#
# # torch.onnx.export(resnet50, dummy_input, "./resnet50_pytorch.onnx", verbose=False) #batch 32
#
# model = mlp()
#
# #####todo 1. After pre-trained load the model
# model.load_state_dict(torch.load('./before.pth'))
# model.eval()
# torch.onnx.export(model, dummy_input, "./before.onnx", verbose=False)
# print('done1')
# model = mlp()
# model.load_state_dict(torch.load('./after.pth'))
# model.eval()
# torch.onnx.export(model, dummy_input, "./after.onnx", verbose=False)
# print('done')