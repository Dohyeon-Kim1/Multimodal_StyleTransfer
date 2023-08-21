import gc
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


###########
## Memory
###########

def empty_memory():
    
  torch.cuda.empty_cache()
  gc.collect()

########
## I/O 
########
  
def print_highlight(text):

  print("="*len(text))
  print(text)
  print("="*len(text)+"\n")

def str_input(text):

  text = input(text)
  print()

  return text

def array_input(text, dtype):

  arr = np.array(input(text).split(","), dtype=dtype)
  print()

  return arr

#################
## Image & Mask
#################

def show_image(image, axis="on"):

  plt.imshow(image)
  plt.axis(axis)
  plt.show()
  print()
  time.sleep(0.5)

def show_masks(image, masks, axis="on", only_arr=False, fix_color=False):

  image = np.array(image) / 255
  mask_color = np.array([[1,0,0],[0,1,0],[0,0,1]])
  choose = 1

  for mask in masks:
    if not(only_arr):
      mask = mask['segmentation']
    if not(fix_color):
      choose = np.random.randint(3)
    mask = np.transpose(np.stack([mask,mask,mask], axis=0), (1,2,0))
    image[mask] = 0.7*image[mask] + 0.3*(np.ones_like(mask)*mask_color[choose])[mask]
  image = Image.fromarray((image*255).astype("uint8"))

  plt.imshow(image)
  plt.axis(axis)
  plt.show()
  print()
  time.sleep(0.5)

def show_point_mask(image, point, mask, axis="on"):

  image = np.array(image) / 255
  mask = np.transpose(np.stack([mask,mask,mask], axis=0), (1,2,0))
  h, w = image.shape[:2]
  x, y = int(point[0]), int(point[1])

  mask_color = np.array([0,1,0])
  image[mask] = 0.7*image[mask] + 0.3*(np.ones_like(mask)*mask_color)[mask]

  dot_color = np.array([1,0,0])
  for i in range(max(0,x-7),min(w,x+8)):
    image[max(0,y-1):min(h,y+2),i] = dot_color
  for j in range(max(0,y-7),min(h,y+8)):
    image[j,max(0,x-1):min(w,x+2),:] = dot_color

  image = Image.fromarray((image*255).astype("uint8"))

  plt.imshow(image)
  plt.axis(axis)
  plt.show()
  print()
  time.sleep(0.5)

def show_image_mask_pairs(content_image, pairs):

  content_image = np.array(content_image)

  plt.figure()
  idx = 1

  for mask, image in pairs:

    if image is None:
      image = content_image.copy()
    
    plt.subplot(len(pairs),2,idx)
    plt.imshow(Image.fromarray(content_image*np.stack([mask,mask,mask], axis=2)))
    plt.axis("off")
    idx += 1

    plt.subplot(len(pairs),2,idx)
    plt.imshow(image)
    plt.axis("off")
    idx += 1

  plt.show()
  print()
  time.sleep(0.5)

def mask_by_point(masks, point):

  for mask in masks:
    if mask['segmentation'][point[1],point[0]]:
      selected_mask = mask['segmentation']
      return selected_mask

  return None

def merge_masks(masks):

  merged_mask = np.zeros_like(masks[0], dtype=bool)
  for mask in masks:
    merged_mask = merged_mask | mask

  return merged_mask
