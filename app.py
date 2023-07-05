import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

st.title("Object & Background Style Transfer with Text")
  
import os
import requests
import torch
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

def masked_image(image, point, mask):
  
  image = np.array(image) / 255
  mask = np.stack([mask,mask,mask], axis=2)
  h, w = image.shape[:2]

  mask_color = np.array([0,1,0])
  image[mask] = 0.7*image[mask] + 0.3*(np.ones_like(mask)*mask_color)[mask]

  if point is not None:
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

if "models" not in st.session_state:

  ## models
  from diffusers import StableDiffusionPipeline
  from models.SAM.segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
  from models.AdaIN.inference import StyleTransfer

  ## load models
  st.session_state.stable_diffusion = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16)
  st.session_state.sam = sam_model_registry["vit_h"](checkpoint="models/SAM/segment_anything/model_zoo/sam_vit_h.pth")
  st.session_state.adain = StyleTransfer(enc_path="models/AdaIN/model_zoo/encoder.pth", dec_path="models/AdaIN/model_zoo/decoder.pth")
  st.session_state.stable_diffusion.to("cuda")
  st.session_state.sam.to("cuda")
  st.session_state.adain.to("cuda")
  st.session_state.sam_manual = SamPredictor(st.session_state.sam)
  st.session_state.models = True

if "step1" not in st.session_state:
  st.session_state.step1 = False
if "step2" not in st.session_state:
  st.session_state.step2 = False
if "step3" not in st.session_state:
  st.session_state.step3 = False
if "get_mask" not in st.session_state:
  st.session_state.get_mask = True

if "content_image" not in st.session_state:
  st.session_state.content_image = None
if "style_image" not in st.session_state:
  st.session_state.style_image = None
if "new_image" not in st.session_state:
  st.session_state.new_image = None
if "merged_mask" not in st.session_state:
  st.session_state.merged_mask = None
if "pairs" not in st.session_state:
  st.session_state.pairs = []


############################################
###### Step1
############################################

if not(st.session_state.step1):
  st.subheader("Step1 : Content Image")  
  button_next = st.button("next", key=0)
  if button_next:
    if st.session_state.content_image is None:
      st.error("To continue next step, you should make or load content image.")
    else:
      st.session_state.step1 = True
      st.experimental_rerun()

if not(st.session_state.step1):

  ## select how to load image
  option = st.radio("Select how to load the content iamge",("url","path","create"))

  ## load image from url
  if option == "url":

    url = st.text_input("Enter the url:")
    button = st.button("load", key=1)
    if button:
      image = Image.open(requests.get(url, stream=True).raw)
    else:
      st.stop()

  ## load image from path
  elif option == "path":

    path = st.text_input("Enter the path:\n")
    button = st.button("load", key=1)
    if button:
      image = Image.open(path)
    else:
      st.stop()

  ## create image
  elif option == "create":

    text = st.text_input("Enter text description of the image which you want to create:")
    guidance_scale = st.slider("Select guidance scale value", 2.5, 7.5, 5.0)
    diffusion_step = st.slider("Select diffusion step value", 25, 100, 25)
    button = st.button("create", key=1)
    if button:
      image =  st.session_state.stable_diffusion(text, guidance_scale=guidance_scale, num_inference_steps=diffusion_step).images[0]
    else:
      st.stop()

  ## resize image for memory when either width or height is larger than 512
  w, h = image.size
  if max(w,h) > 512:
    resize_scale = 512 / max(w,h)
    image = image.resize((int(w*resize_scale), int(h*resize_scale)))

  ## show image
  st.write("Content Image")
  st.image(image)
  st.session_state.content_image = image
  st.write("If you are satisfied with above image, click 'next' button.")

############################################
###### Step2
############################################

if st.session_state.step1 and not(st.session_state.step2):

  st.subheader("Step2 : Mask & Style Image Pairs")
  button_prev = st.button("prev", key=0)
  button_next = st.button("next", key=1)

  if button_prev:
    st.session_state.step1 = False
    st.session_state.content_image = None
    st.session_state.style_image = None
    st.session_state.merged_mask = None
    st.session_state.pairs = []
    st.experimental_rerun()
  
  if button_next:
    if len(st.session_state.pairs) == 0:
      st.error("To continue next step, you should mask & image pairs")
    else:
      st.session_state.step2 = True
      st.experimental_rerun()
  
if st.session_state.step1 and not(st.session_state.step2):
  
  if st.session_state.get_mask:

    st.write("click 'reset' button if you want to reset mask & style image pair.")
    button_new = st.button("reset", key=3)
    if button_new:
      st.session_state.pairs = []
      st.experimental_rerun()

    st.write("click 'background' button if you want to background mask & style image pair")
    button_background = st.button("background")
    if button_background:
      if len(st.session_state.pairs) == 0:
        st.session_state.merged_mask = np.ones_like(np.array(st.session_state.content_image)[:,:,0], dtype=bool)
      else:
        st.session_state.merged_mask = (merge_masks([pair[0] for pair in st.session_state.pairs]) == False)
      st.session_state.get_mask = False
      st.session_state.style_image = None
      st.experimental_rerun()

    st.session_state.sam_manual.set_image(np.array(st.session_state.content_image))
    st.write("click the part of which you want to get mask")
    coordinates = streamlit_image_coordinates(np.array(st.session_state.content_image))
    
    if coordinates is not None:
      
      coordinates = np.array([[coordinates["x"],coordinates["y"]]])
      selected_mask, _, _ = st.session_state.sam_manual.predict(point_coords=coordinates, point_labels=np.array([1]), multimask_output=False)
      selected_mask = np.squeeze(selected_mask, axis=0)
      new_image = masked_image(st.session_state.content_image, coordinates[0], selected_mask)
      st.write("Selected Mask")
      st.image(new_image)

      st.write("click 'add' button to add selected mask")
      button_add = st.button("add", key=4)
      if button_add:
        if st.session_state.merged_mask is None:
          st.session_state.merged_mask = selected_mask
        else:
          st.session_state.merged_mask = merge_masks([st.session_state.merged_mask, selected_mask])
      
    if st.session_state.merged_mask is not None:

      st.write("Merged Mask")
      st.image(masked_image(st.session_state.content_image, None, st.session_state.merged_mask))

      button_style = st.button("go to select style image", key=5)
      if button_style:
        st.session_state.get_mask = False
        st.session_state.style_image = None
        st.experimental_rerun()
    
    if len(st.session_state.pairs) != 0:

      st.write("Created Mask & Style Image Pairs")
      for mask, style_image in st.session_state.pairs:
        col1, col2 = st.columns([1,1])
        with col1:
          st.image(masked_image(st.session_state.content_image, None, mask))
        with col2:
          st.image(style_image)
    
  if not(st.session_state.get_mask):
    
    st.write("click 'select image' button if you finish selecting image")
    button_select = st.button("select image", key=4)
    if button_select:
      if st.session_state.style_image is None:
        st.error("You should select style image")
      else:
        st.session_state.get_mask = True
        st.session_state.pairs.append([st.session_state.merged_mask, st.session_state.style_image])
        st.session_state.merged_mask = None
        st.experimental_rerun()

    ## select how to load image
    option = st.radio("Select how to load the style iamge",("url","path","create","no"))

    ## load image from url
    if option == "url":

      url = st.text_input("Enter the url:")
      button = st.button("load", key=5)
      if button:
        image = Image.open(requests.get(url, stream=True).raw)
      else:
        st.stop()

    ## load image from path
    elif option == "path":

      path = st.text_input("Enter the path:\n")
      button = st.button("load", key=5)
      if button:
        image = Image.open(path)
      else:
        st.stop()

    ## create image
    elif option == "create":

      text = st.text_input("Enter text description of the image which you want to create:")
      guidance_scale = st.slider("Select guidance scale value", 2.5, 7.5, 5.0)
      diffusion_step = st.slider("Select diffusion step value", 25, 100, 25)
      button = st.button("create", key=5)
      if button:
        image =  st.session_state.stable_diffusion(text, guidance_scale=guidance_scale, num_inference_steps=diffusion_step).images[0]
      else:
        st.stop()
    
    elif option == "no":

      st.write("no use style image")
      button = st.button("confirm", key=5)
      if button:
        image = st.session_state.content_image.copy()
      else:
        st.stop()

    ## resize image for memory when either width or height is larger than 512
    w, h = image.size
    if max(w,h) > 512:
      resize_scale = 512 / max(w,h)
      image = image.resize((int(w*resize_scale), int(h*resize_scale)))

    ## show image
    st.write("Style Image")
    st.image(image)
    st.session_state.style_image = image

############################################
###### Step3
############################################

if st.session_state.step1 and st.session_state.step2:

  st.subheader("Step3 : Style Transfer")
  button_prev = st.button("prev", key=0)

  if button_prev:
    st.session_state.step2 = False
    st.experimental_rerun()

if st.session_state.step1 and st.session_state.step2:

  st.write("Style Transfer Options")
  preserve_color = st.checkbox("Peserve color", True)
  alpha = st.slider("alpha", 0.0, 1.0, 1.0)

  button_run = st.button("style transfer", key=1)
  
  if button_run:
    
    new_image = np.zeros_like(np.array(st.session_state.content_image), dtype=np.float32)

    ## create style transfered image
    for mask, style_image in st.session_state.pairs:

      ## not style transfer for the mask part
      if style_image == st.session_state.content_image:
        new_image += (np.array(content_image)/255) * np.stack([mask,mask,mask], axis=2)

      ## style transfer for the mask part
      else:
        transfered_image = st.session_state.adain(np.array(st.session_state.content_image)/255, np.array(style_image)/255, alpha=alpha, preserve_color=preserve_color, device="cuda")
        if transfered_image.shape != new_image.shape:
          transfered_image = cv2.resize(transfered_image, (new_image.shape[1], new_image.shape[0]))
        new_image += transfered_image * np.stack([mask,mask,mask], axis=2)

    ## np.ndarray to PIL.Image
    new_image = Image.fromarray((new_image*255).astype(np.uint8))
    st.session_state.new_image = new_image

  if st.session_state.new_image is not None:
    st.write("Style Transferd Image")
    st.image(st.session_state.new_image)
    buf = BytesIO()
    st.session_state.new_image.save(buf, format="png")
    st.download_button(label="download", data=buf.getvalue(), file_name="new_image.png", mime="image/png")
