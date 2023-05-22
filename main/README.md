
## MN Pruning(intro)

추론 과정에서 사용되는 tensorRT의 set_weight_sparsity옵션은 tensorRT > 8.0.0 에서만 사용 가능합니다.

더 효율적인 네트워크는 제한된 시간 예산으로 더 나은 예측을 하고,
제한된 배포 환경에 적합할 수 있습니다.(ex. mobile(edge device))  

`프루닝(pruning)`은 이러한 목표를 달성할 수 있는 최적화 기술 중 하나입니다. 
네트워크에 가중치가 0으로 구성되있으면, 이를 최적화하기 쉽습니다. 그러나 이를 실현하는 데 세 가지 어려움이 있습니다.<br>
<br>
*Acceleration* - <br>

세분화되고 구조화되지 않은 가중치 희소성은 구조가 부족하고 일반적인 네트워크 작업을 가속화하기 위해 효율적인 하드웨어에서 사용할 수 있는 벡터 및 행렬 명령을 사용할 수 없습니다.   
표준 sparse 형식은 높은 희소성을 제외하고는 모두 비효율적입니다.<br>
<br>
*Accuracy* - <br>

세분화되고 구조화되지 않은 희소성으로 유용한 속도 향상을 달성하려면 네트워크를 희소하게 만들어야 하며, 이로 인해 종종 정확도 손실이 발생합니다.  
    채널을 제거하는(채널 프루닝) 가속을 더 쉽게 하는 다른 가지치기 방법은 정확도 문제가 더 빨리 발생할 수 있습니다.<br>
<br>
*Workflow* - <br>

네트워크 A가 희소성 X를 달성할 수 있지만, 희소성 X를 네트워크 B에 적용하려고 할 때 문제가 발생합니다.  
    네트워크, 작업, 최적화 또는 하이퍼 매개 변수의 차이로 인해 작동하지 않을 수 있습니다.
<br> 
<br>    
이 [Accelerating Sparse DNN 논문](https://arxiv.org/pdf/2104.08378.pdf) 에서는 NVIDIA 아키텍처가 이러한 문제를 해결하는 방법에 대해 설명하고  
NVIDIA는 NVIDIA Ampere 아키텍처 GPU에서 사용할 수 있는 Sparse Tensor 코어에 대한 지원을 소개합니다.<br>
<br>
TensorRT는 고성능 딥 러닝 추론을 위한 SDK(software development kit)로, 대기 시간을 최소화하고 처리량을 최대화하는 옵티마이저 및 런타임을 포함합니다.  
Sparse Tensor Core는 신경망에서 불필요한 계산을 제거하여 밀도가 높은 네트워크에 비해 와트당 성능을 30% 이상 높일 수 있습니다.

- 참고 

[Accelerating Sparse DNN 논문](https://arxiv.org/pdf/2104.08378.pdf)

[Acceleration sparse DNN 기술 블로그](https://moon-walker.medium.com/리뷰-accelerating-sparse-deep-neural-networks-870b88c0e2bc)

## USAGE to prune M:N Sparsity 

```bash
# CIFAR100 / ResNet56(only) - default = M:N Sparsity=True
$ python Cifar100_train_main.py 

# ImageNet / ResNet(50/101/152...)
$ python resnet_training_main.py --MNmode --ONNX=$$$$$.onnx --pretrained=true --arch resnet50
$ python resnet_training_main.py --MNmode --ONNX=$$$$$.onnx --pretrained=true --arch resnet101
$ python resnet_training_main.py --MNmode --ONNX=$$$$$.onnx --pretrained=true --arch resnet152
```

```python
# 기존의 학습용 코드에서 
# 아래의 코드를 train코드 전에 넣어서 프루닝을 하고, fine-tuning을 5epochs 정도로 하여 accuracy를 회복.
from apex.contrib.sparsity import ASP

ASP.prune_trained_model(model, optimizer)

train()
eval()
```

## pruning ratio 구하기(프루닝이 잘 되었는지 확인)

in `MN_check.py`

```python
import torch
import torchvision

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
```

## M:N ratio 구하기(M:N의 비율이 잘 들어갔는지 확인)

```python
PATH = args.pthFile
import_model = torchvision.models.resnet50(weights=None)
load = torch.load(PATH)

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

# Default값은 4:2이므로 4:2가 맞는지 확인.
```

## pth파일의 val_accuracy 구하기

```bash
# 이미지넷
$ python trt_accuracy.py --pthFile $$$$$path$$$$$ --img_path $$$$$path$$$$$
# cifar100
$ python trt_accuracy.py --pthFile $$$$$path$$$$$ --img_path $$$$$path$$$$$ --CIFAR100
```

