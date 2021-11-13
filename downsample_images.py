import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2

from bicubic_interpolation import bicubic


inPath = os.getcwd() + '\\Images'
outPath = os.getcwd() + '\\Images_DownSampled'
for imagePath in os.listdir(inPath):
    inputPath = os.path.join(inPath, imagePath)
    img = cv2.imread(inputPath, cv2.IMREAD_COLOR)
    img_downsamp = bicubic(img, 0.25, -1/2)
    split = imagePath.split(".")
    newfileName = split[0] + '_down_4x.'+ split[1]
    fullOutPath = os.path.join(outPath, newfileName)
    cv2.imwrite(fullOutPath, img_downsamp)