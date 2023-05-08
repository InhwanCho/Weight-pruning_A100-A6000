import torchvision.models as models
import torch
import torch.onnx

# load the pretrained model
resnet50 = models.resnet50(pretrained=True, progress=False).eval()

BATCH_SIZE=32

dummy_input=torch.randn(BATCH_SIZE, 3, 224, 224)

# export the model to ONNX
torch.onnx.export(resnet50, dummy_input, "resnet50_pytorch.onnx", verbose=False)

from skimage import io
from skimage.transform import resize
# from matplotlib import pyplot as plt
import numpy as np

print(torch.__version__)
print(torch.cuda.is_available())
url='https://images.dog.ceo/breeds/retriever-golden/n02099601_3004.jpg'
img = resize(io.imread(url), (224, 224))
img = np.expand_dims(np.array(img, dtype=np.float32), axis=0) # Expand image to have a batch dimension
input_batch = np.array(np.repeat(img, BATCH_SIZE, axis=0), dtype=np.float32) # Repeat across the batch dimension

resnet50_gpu = models.resnet50(pretrained=True, progress=False).to("cuda").eval()

input_batch_chw = torch.from_numpy(input_batch).transpose(1,3).transpose(2,3)
input_batch_gpu = input_batch_chw.to("cuda")

input_batch_gpu.shape


with torch.no_grad():
    predictions = np.array(resnet50_gpu(input_batch_gpu).cpu())

resnet50_gpu_half = resnet50_gpu.half()
input_half = input_batch_gpu.half()

indices = (-predictions[0]).argsort()[:5]
print("Class | Likelihood")
list(zip(indices, predictions[0][indices]))

import os

os._exit(0) # Shut down all kernels so TRT doesn't fight with PyTorch for GPU memory

BATCH_SIZE = 32

import numpy as np

USE_FP16 = True
target_dtype = np.float16 if USE_FP16 else np.float32

from skimage import io
from skimage.transform import resize
from matplotlib import pyplot as plt
import numpy as np

url='https://images.dog.ceo/breeds/retriever-golden/n02099601_3004.jpg'
img = resize(io.imread(url), (224, 224))
input_batch = np.array(np.repeat(np.expand_dims(np.array(img, dtype=np.float32), axis=0), BATCH_SIZE, axis=0), dtype=np.float32)

import torch
from torchvision.transforms import Normalize

def preprocess_image(img):
    norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    result = norm(torch.from_numpy(img).transpose(0,2).transpose(1,2))
    return np.array(result, dtype=np.float16)

preprocessed_images = np.array([preprocess_image(image) for image in input_batch])


import tensorrt
#
# # step out of Python for a moment to convert the ONNX model to a TRT engine using trtexec
# if USE_FP16:
#     !trtexec --onnx=resnet50_pytorch.onnx --saveEngine=resnet_engine_pytorch.trt  --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16
# else:
#     !trtexec --onnx=resnet50_pytorch.onnx --saveEngine=resnet_engine_pytorch.trt  --explicitBatch
#
# import numpy as np
#
# # need to set input and output precisions to FP16 to fully enable it
# output = np.empty([BATCH_SIZE, 1000], dtype = target_dtype)
#
# # allocate device memory
# d_input = cuda.mem_alloc(1 * input_batch.nbytes)
# d_output = cuda.mem_alloc(1 * output.nbytes)
#
# bindings = [int(d_input), int(d_output)]
#
# stream = cuda.Stream()
#
#
# def predict(batch):  # result gets copied into output
#     # transfer input data to device
#     cuda.memcpy_htod_async(d_input, batch, stream)
#     # execute model
#     context.execute_async_v2(bindings, stream.handle, None)
#     # transfer predictions back
#     cuda.memcpy_dtoh_async(output, d_output, stream)
#     # syncronize threads
#     stream.synchronize()
#
#     return output
#
# print("Warming up...")
#
# pred = predict(preprocessed_images)
#
# print("Done warming up!")
#
# pred = predict(preprocessed_images)
#
# indices = (-pred[0]).argsort()[:5]
# print("Class | Probability (out of 1)")
# list(zip(indices, pred[0][indices]))