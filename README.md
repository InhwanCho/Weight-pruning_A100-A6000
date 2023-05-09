## Env setting
```
$ docker pull whdlsghks/a100_a6000:1.0
$ git clone https://github.com/InhwanCho/Weight-pruning_A100-A6000.git
#$ pip install -r requirements.txt
$ cd main/
```

## to prune M:N Sparsity
```

# CIFAR100 / ResNet56(only) - default = M:N Sparsity=True
$ python Cifar100_train_main.py 

# ImageNet / ResNet(50/101/152...)
$ python resnet_training_main.py --MNmode --ONNX=$$$$$.onnx --pretrained=true --arch resnet50
$ python resnet_training_main.py --MNmode --ONNX=$$$$$.onnx --pretrained=true --arch resnet101
$ python resnet_training_main.py --MNmode --ONNX=$$$$$.onnx --pretrained=true --arch resnet152
```

## to convert onnx to trt_engine
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
# set the sh file
$ sed -i -e 's/\r$//' auto_python.sh
$ vim auto_python.sh
```

- caution
when you measure inference time, you are not supposed to use sh file and you should execute `trt_inference.py`file with some break time
