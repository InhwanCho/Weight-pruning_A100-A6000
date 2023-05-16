#sed -i -e 's/\r$//' auto_python.sh
#
echo 'hello, it starts to MN prune'
# MN Pruning using pretrained_file in only 1 epoch
python resnet_training_main.py --MNmode --ONNX=imagenet_res50_pruned.onnx --pretrained=true --arch resnet50 --epochs 1
#python resnet_training_main.py --MNmode --ONNX=imagenet_res101_pruned.onnx --pretrained=true --arch resnet101 --epochs 1
#python resnet_training_main.py --MNmode --ONNX=imagenet_res152_pruned.onnx --pretrained=true --arch resnet152 --epochs 1
#
echo 'start to build trt_engine(onnx to trt)'
# build trt engine should set (optimal batch_size),(max_batch),(sparsity)
python onnx2trt.py --trtFile=tem_folder/imagenet_res50_pruned.trt --onnxFile=tem_folder/imagenet_res50_pruned.onnx --opt_batch 32 --max_batch=256 --Sparsity
#
echo 'testing trt engine accuracy'
python trt_accuracy.py --trtFile=tem_folder/imagenet_res50_pruned.trt
#python trt_inference.py --trtFile=tem_folder/imagenet_res50_pruned.trt --batch_size=16 --Sparsity
#
echo 'measure the latency using torch event, shoud set the batch_size==opt_batch'
echo 'never measure the latency using shell script, this one is just sample'
python trt_inference.py --trtFile=tem_folder/imagenet_res50_pruned.trt --batch_size=32 --Sparsity
echo 'finish the work'
#
#if you need to make lots of trt_engines, samples are blow
#CUDA_MODULE_LOADING=LAZY python onnx2trt.py --trtFile=tem_folder/imagenet_res152_pruned_1F_TF32.trt --onnxFile=tem_folder/imagenet_res152_pruned.onnx --opt_batch=1 --max_batch=256 --NotUseFP16Mode
#CUDA_MODULE_LOADING=LAZY python onnx2trt.py --trtFile=tem_folder/imagenet_res152_pruned_16S_TF32.trt --onnxFile=tem_folder/imagenet_res152_pruned.onnx --opt_batch=16 --max_batch=256 --Sparsity --NotUseFP16Mode
#CUDA_MODULE_LOADING=LAZY python onnx2trt.py --trtFile=tem_folder/imagenet_res152_pruned_16F_TF32.trt --onnxFile=tem_folder/imagenet_res152_pruned.onnx --opt_batch=16 --max_batch=256 --NotUseFP16Mode
#CUDA_MODULE_LOADING=LAZY python onnx2trt.py --trtFile=tem_folder/imagenet_res152_pruned_64S_TF32.trt --onnxFile=tem_folder/imagenet_res152_pruned.onnx --opt_batch=64 --max_batch=256 --Sparsity --NotUseFP16Mode
#CUDA_MODULE_LOADING=LAZY python onnx2trt.py --trtFile=tem_folder/imagenet_res152_pruned_64F_TF32.trt --onnxFile=tem_folder/imagenet_res152_pruned.onnx --opt_batch=64 --max_batch=256 --NotUseFP16Mode
#CUDA_MODULE_LOADING=LAZY python onnx2trt.py --trtFile=tem_folder/imagenet_res152_pruned_128S_TF32.trt --onnxFile=tem_folder/imagenet_res152_pruned.onnx --opt_batch=128 --max_batch=256 --Sparsity --NotUseFP16Mode
#CUDA_MODULE_LOADING=LAZY python onnx2trt.py --trtFile=tem_folder/imagenet_res152_pruned_128F_TF32.trt --onnxFile=tem_folder/imagenet_res152_pruned.onnx --opt_batch=128 --max_batch=256 --NotUseFP16Mode
#CUDA_MODULE_LOADING=LAZY python onnx2trt.py --trtFile=tem_folder/imagenet_res152_pruned_256S_TF32.trt --onnxFile=tem_folder/imagenet_res152_pruned.onnx --opt_batch=256 --max_batch=256 --Sparsity --NotUseFP16Mode
#CUDA_MODULE_LOADING=LAZY python onnx2trt.py --trtFile=tem_folder/imagenet_res152_pruned_256F_TF32.trt --onnxFile=tem_folder/imagenet_res152_pruned.onnx --opt_batch=256 --max_batch=256 --NotUseFP16Mode
