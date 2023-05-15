# Getting Started

To get started with weight_pruning_A100-A6000, follow these steps.

## Step 1. Install Dependencies

```
$ docker pull whdlsghks/a100_a6000:1.0
#$ docker run ...
$ git clone https://github.com/InhwanCho/Weight-pruning_A100-A6000.git
$ cd main/

# or

$ git clone https://github.com/InhwanCho/Weight-pruning_A100-A6000.git
$ cd main/
$ docker pull nvcr.io/nvidia/tensorrt:23.03-py3
#$ docker run ...
$ pip install -r requirements.txt
```

## Step 2. Pruning M:N Sparsity
```
# please change the folder&file_name first
# origin code : [https://github.com/pytorch/examples/tree/main/imagenet]

$ python Cifar100_train_main.py 

# ImageNet / ResNet(50/101/152...)
$ python resnet_training_main.py --MNmode --ONNX=$$$$$.onnx --pretrained=true --arch resnet50
$ python resnet_training_main.py --MNmode --ONNX=$$$$$.onnx --pretrained=true --arch resnet101
$ python resnet_training_main.py --MNmode --ONNX=$$$$$.onnx --pretrained=true --arch resnet152

#CIFAR100 / ResNet56(only)
$ python CIFAR100_training_main.py
```

## Step 3. Converting onnx to trt_engine
```
# build engine
# if you are using CIFAR100 put '--CIFAR100' option
$ python onnx2trt.py --trtFile=$$$$$.trt --onnxFile=$$$$$.onnx --max_batch=256 --Sparsity

# checking the accuracy
$ python trt_accuracy.py --trtFile=$$$$$.trt

# measure the latency
$ python trt_inference.py --trtFile=$$$$$.trt --batch_size=1 --Sparsity
```

### to convert many trt_engine
```
# set the sh file(sample)
$ sed -i -e 's/\r$//' auto_python.sh
$ vim auto_python.sh
```

## Usage for onnx2trt & trt_accuracy & trt_inference
```
    python onnx2trt.py [--trtFile trtfile] [--onnxFile onnxfile]
             [--pthFile pthfile] [--batch_size batchsize_for_inference]
             [--opt_batch optimal_batchsize_for_set_trtengine_shaep]
             [--maxbatch max_batchsize_for_set_trtengine_shaep]
             [--Sparsity] # Sparsity Enable option default=True
             [--NotUseFP16Mode] # Convert TF32 to FP16
             [--CIFAR100] # dataset=CIFAR100/ default = IMAGENET
```


- caution
when you measure inference time, you are not supposed to use sh file and you should execute `trt_inference.py`file with some break time
<br>
when you have CUDA version matching error, please import torch after import tensorrt(build/ load engine first)
<br>
<br>

<details>
<summary>#show the result tables and summary#</summary>
<div markdown="1">

A100 table
![캡처](src/a100_imgnet.PNG)
![캡처](src/a100_cifar100.PNG)
<br>
<br>
RTX-A6000 table
![캡처](src/a6000_imagenet.PNG)
![캡처](src/a6000_cifar100.PNG)
<br>

</div>
</details>

<details>
<summary>#FP16 Summary#</summary>
<div markdown="1">
Summary graph(line chart, FP16) <br>
![캡처](src/graph.PNG)
    
<br>    

GPU(A100, A6000)별, batch_size에 따른 trt engine의 추론 속도 차이[위]/변화율[아래]<br>
각 실험 당 optimal_batch를 지정하여 engine을 만들어서 실험을 진행<br>
(각 실험 당 1개의 trt_file 생성)<br>
ResNet56은 CIFAR-100 dataset을 사용.<br>
추론 속도가 batch=256 이여도 빠르기 때문인지, 추론 속도가 batch=1에서만 약 8% 증가<br>
ResNet50, 101, 152는 ImageNet을 사용하였고,<br>
Set Sparsity weight = True 하였을 때 전반적으로 속도가 감소.<br>
[위의 2개의 그래프 A100, 6000 / inference time by batch size]<br>
ResNet50, 101, 152는 batch=1에서는 증가율이 적으나, <br>
batch=16 이상의 경우 일반적으로 높은 증가율을 보임<br>
[아래 2개의 그래프 A100, 6000 / speed increase rate]<br>


</div>
</details>


<details>
<summary>#TF32 Summary#</summary>
<div markdown="1">

Summary(line chart, TF32)

ResNet152(ImageNet)에서만 실험(FP16에서 시간 차이가 가장 컸기때문에)<br>
TF32모드에서는 A100에서는 inference time의 증가가 거의 없고, <br>
RTX-A6000에서는 batch_size=1일때만 약 8% 증가

tf32 table

![캡처](src/tf32.PNG)

</div>
</details>

<details>
<summary>### Refences ###</summary>
<div markdown="1">
    
[Notion link in detail](https://www.notion.so/keti-via/NPU-Weight-pruning-A100-A6000-Latency-2518e742b26e47e88b79ed9abac98166)

M:N sparsity Technical blog, NVIDIA 공식 문서 1,공식 문서 2 

[Accelerating Inference with Sparsity Using the NVIDIA Ampere Architecture and NVIDIA TensorRT | NVIDIA Technical Blog](https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/)

[NVIDIA tensorRT](https://docs.nvidia.com/deeplearning/tensorrt/api/index.html)

[IExecutionContext — NVIDIA TensorRT Standard Python API Documentation 8.6.0 documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/ExecutionContext.html#tensorrt.IExecutionContext.execute_async_v3)


- 코드 참고용 tnsorRT, 최신 버전 TensorRT 예제, MN spartsity(pruning)

[NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT)

[NVIDIA/trt-samples-for-hackathon-cn](https://github.com/NVIDIA/trt-samples-for-hackathon-cn)

[NVIDIA/apex](https://github.com/NVIDIA/apex)

- NM-sparsity/ trt엔진 상세 분석/ 논문리뷰

[TensorRT 코드 참고](https://github.com/aojunzz/NM-sparsity)
    
[TensorRT 분석](https://velog.io/@sjj995/TensorRT-Polygraphy를-활용하여-간단하게-trt-engine-추론-과정-알아보기)

[Acceleration sparse DNN 논문 리뷰](https://moon-walker.medium.com/리뷰-accelerating-sparse-deep-neural-networks-870b88c0e2bc)
    
</div>
</details>
