#sed -i -e 's/\r$//' auto_python.sh
echo 'hello'
#comment
#python resnet_training_main.py --MNmode --ONNX=imagenet_res50_pruned.onnx --pretrained=true --arch resnet50
#python resnet_training_main.py --MNmode --ONNX=imagenet_res101_pruned.onnx --pretrained=true --arch resnet101
#python resnet_training_main.py --MNmode --ONNX=imagenet_res152_pruned.onnx --pretrained=true --arch resnet152
echo 'start the work'
#python onnx2trt.py --trtFile=tem_folder/imagenet_res50_pruned.trt --onnxFile=tem_folder/imagenet_res50_pruned.onnx --max_batch=256 --Sparsity
#python trt_accuracy.py --trtFile=tem_folder/imagenet_res101_pruned_1F.trt
#python trt_inference.py --trtFile=tem_folder/imagenet_res50_pruned.trt --batch_size=16 --Sparsity
CUDA_MODULE_LOADING=LAZY python onnx2trt.py --trtFile=tem_folder/imagenet_res152_pruned_1S_TF32.trt --onnxFile=tem_folder/imagenet_res152_pruned.onnx --opt_batch=1 --max_batch=256 --NotUseFP16Mode --Sparsity
CUDA_MODULE_LOADING=LAZY python onnx2trt.py --trtFile=tem_folder/imagenet_res152_pruned_1F_TF32.trt --onnxFile=tem_folder/imagenet_res152_pruned.onnx --opt_batch=1 --max_batch=256 --NotUseFP16Mode
CUDA_MODULE_LOADING=LAZY python onnx2trt.py --trtFile=tem_folder/imagenet_res152_pruned_16S_TF32.trt --onnxFile=tem_folder/imagenet_res152_pruned.onnx --opt_batch=16 --max_batch=256 --Sparsity --NotUseFP16Mode
CUDA_MODULE_LOADING=LAZY python onnx2trt.py --trtFile=tem_folder/imagenet_res152_pruned_16F_TF32.trt --onnxFile=tem_folder/imagenet_res152_pruned.onnx --opt_batch=16 --max_batch=256 --NotUseFP16Mode
CUDA_MODULE_LOADING=LAZY python onnx2trt.py --trtFile=tem_folder/imagenet_res152_pruned_64S_TF32.trt --onnxFile=tem_folder/imagenet_res152_pruned.onnx --opt_batch=64 --max_batch=256 --Sparsity --NotUseFP16Mode
CUDA_MODULE_LOADING=LAZY python onnx2trt.py --trtFile=tem_folder/imagenet_res152_pruned_64F_TF32.trt --onnxFile=tem_folder/imagenet_res152_pruned.onnx --opt_batch=64 --max_batch=256 --NotUseFP16Mode
CUDA_MODULE_LOADING=LAZY python onnx2trt.py --trtFile=tem_folder/imagenet_res152_pruned_128S_TF32.trt --onnxFile=tem_folder/imagenet_res152_pruned.onnx --opt_batch=128 --max_batch=256 --Sparsity --NotUseFP16Mode
CUDA_MODULE_LOADING=LAZY python onnx2trt.py --trtFile=tem_folder/imagenet_res152_pruned_128F_TF32.trt --onnxFile=tem_folder/imagenet_res152_pruned.onnx --opt_batch=128 --max_batch=256 --NotUseFP16Mode
CUDA_MODULE_LOADING=LAZY python onnx2trt.py --trtFile=tem_folder/imagenet_res152_pruned_256S_TF32.trt --onnxFile=tem_folder/imagenet_res152_pruned.onnx --opt_batch=256 --max_batch=256 --Sparsity --NotUseFP16Mode
CUDA_MODULE_LOADING=LAZY python onnx2trt.py --trtFile=tem_folder/imagenet_res152_pruned_256F_TF32.trt --onnxFile=tem_folder/imagenet_res152_pruned.onnx --opt_batch=256 --max_batch=256 --NotUseFP16Mode
echo 'finish the work'

#python trt_inference.py --trtFile=tem_folder/imagenet_res152_pruned_1S_TF32.trt --batch_size=1 --Sparsity