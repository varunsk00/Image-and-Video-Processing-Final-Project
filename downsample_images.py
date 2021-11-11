import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from bicubic_downsample import build_filter, apply_bicubic_downsample

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


def downsampleImage(img, factor):
    tensor = tf.convert_to_tensor(img)
    tensor = tf.expand_dims(tensor,0)
    tensor_flt = tf.cast(tensor, tf.float32)/255

    # First, create the bicubic kernel. This can be reused in multiple downsample operations
    k = build_filter(factor=factor)

    # Downsample x which is a tensor with shape [N, H, W, 3]
    y = apply_bicubic_downsample(tensor_flt, filter=k, factor=factor)

    # y now contains x downsampled to [N, H/factor, W/factor, 3]
    downsamp = tensor_to_image(y)
    return downsamp

inPath = os.getcwd() + '\\Images'
outPath = os.getcwd() + '\\Images_DownSampled'
for imagePath in os.listdir(inPath):
    inputPath = os.path.join(inPath, imagePath)
    img = Image.open(inputPath)
    img_downsamp = downsampleImage(img, 4)
    split = imagePath.split(".")
    newfileName = split[0] + '_down_4x.'+ split[1]
    fullOutPath = os.path.join(outPath, newfileName)
    img_downsamp.save(fullOutPath)