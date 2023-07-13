import gc
import numpy as np
import torch
from PIL import Image

def empty_memory():
  torch.cuda.empty_cache()
  gc.collect()

def masked_image(image, mask, point=None):
  image = np.array(image) / 255
  mask = np.stack([mask,mask,mask], axis=2)

  mask_color = np.array([0,1,0])
  image[mask] = 0.7*image[mask] + 0.3*(np.ones_like(mask)*mask_color)[mask]

  if point is not None:
    h, w = image.shape[:2]
    x, y = int(point[0]), int(point[1])
    dot_color = np.array([1,0,0])
    for i in range(max(0,x-7),min(w,x+8)):
      image[max(0,y-1):min(h,y+2),i] = dot_color
    for j in range(max(0,y-7),min(h,y+8)):
      image[j,max(0,x-1):min(w,x+2),:] = dot_color

  image = Image.fromarray((image*255).astype("uint8"))

  return image

def merge_masks(masks):
  merged_mask = np.zeros_like(masks[0], dtype=bool)
  for mask in masks:
    merged_mask = merged_mask | mask

  return merged_mask
