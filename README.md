# Getting Started

To get started with weight_pruning_A100-A6000, follow these steps.

## Step 1.Install Dependencies

```
$ docker pull whdlsghks/a100_a6000:1.0
$ git clone https://github.com/InhwanCho/Weight-pruning_A100-A6000.git
$ cd main/

#or
$ git clone https://github.com/InhwanCho/Weight-pruning_A100-A6000.git
$ cd main/
$ docker pull nvidia...
$ pip install -r requirements.txt
```

## Step 2. Pruning M:N Sparsity
```

# CIFAR100 / ResNet56(only) - default = M:N Sparsity=True
$ python Cifar100_train_main.py 

# ImageNet / ResNet(50/101/152...)
$ python resnet_training_main.py --MNmode --ONNX=$$$$$.onnx --pretrained=true --arch resnet50
$ python resnet_training_main.py --MNmode --ONNX=$$$$$.onnx --pretrained=true --arch resnet101
$ python resnet_training_main.py --MNmode --ONNX=$$$$$.onnx --pretrained=true --arch resnet152
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

## to convert many trt_engine
```
# set the sh file(sample)
$ sed -i -e 's/\r$//' auto_python.sh
$ vim auto_python.sh
```

## Usage for onnx2trt & trtaccuracy & trtinference
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


<details>
<summary>show the result tables and summary</summary>
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

summary graph
![캡처](src/graph.PNG)

<br>

tf32 table
![캡처](src/tf32.PNG)


</div>
</details>

[link in detail] : <https://www.notion.so/keti-via/NPU-Weight-pruning-A100-A6000-Latency-2518e742b26e47e88b79ed9abac98166>
