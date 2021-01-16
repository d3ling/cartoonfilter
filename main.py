# Cartoon Filter uses Mask R-CNN to detect people and CartooGAN to cartoonize them

# common libraries
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torchvision
import numpy as np
import os
import cv2
import random

# custom
from MaskRCNN.getDetections import getDetections
from CartoonGAN.CartoonGAN import Generator
from CartoonGAN.generateCartoon import generateCartoon

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_img_path = 'input.jpg'
output_img_path = 'output.jpg'
mrcnn_model_path = 'MaskRCNN/pretrained'
cgan_model_path = 'CartoonGAN/pretrained'
styles = ['Hayao', 'Hosoda', 'Paprika', 'Shinkai']

# pretrained Mask R-CNN model
# model originally obtained via torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
mrcnn_model = torch.load(os.path.join(mrcnn_model_path, 'maskrcnn.pth')) # full model with weights
mrcnn_model.eval()

# pretrained CartoonGAN models for all styles
cgan_models = list()
for i in range(len(styles)):
  cgan_models.append(Generator(3, 3, 64, 8)) # Generator(input channels, output channels, channels in hidden layer 1, residual blocks)
  cgan_models[i].load_state_dict(torch.load(os.path.join(cgan_model_path, 'cartoongan_' + styles[i] + '.pth')))
  cgan_models[i].eval()

# load input image as a numpy array
input_img = Image.open(input_img_path)
input_img = np.asarray(input_img)
input_img = input_img.copy()

# obtain masks, bounding boxes, and classes for detected object instances above a confidence level
confidence = 0.7
with torch.no_grad():
  masks, boxes, classes = getDetections(mrcnn_model, input_img, confidence, device)

# filter masks and boxes to contain persons only
masks = masks[classes=='person']
boxes = boxes[classes=='person']

output_img = input_img.copy()

dc_factor = 4 # resizing factor that occurs due to down convolutions in CartoonGAN
for i in range(len(boxes)):
  # coordinates for cropping
  x0, y0 = boxes[i][0]
  x1, y1 = boxes[i][1]
  
  # obtain cropped image and mask
  crop_slice = slice(y0, y1+1), slice(x0, x1+1) # same as y0:y1+1, x0:x1+1
  cropped_img = input_img[crop_slice]
  cropped_mask = masks[i][crop_slice]

  # cartoonize cropped image
  cartoon_img = generateCartoon(cgan_models[i % len(cgan_models)], cropped_img, dc_factor, device)

  # cartoonized objects are extracted using the cropped mask for boolean indexing
  # the corresponding objects in the output image are replaced after cropping and boolean indexing
  output_img[crop_slice][cropped_mask, :] = cartoon_img[cropped_mask, :]

output_img = Image.fromarray(output_img)
output_img.save(output_img_path)