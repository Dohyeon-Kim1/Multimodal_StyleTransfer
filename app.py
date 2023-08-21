%%writefile app.py

import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from utils.utils_app import empty_memory, masked_image, merge_masks

st.title("Make Your Own Image!")

import os
import gc
import requests
import torch
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

## session state setting
if "models" not in st.session_state:
  st.subheader("Loading Models ..")

  from diffusers import StableDiffusionPipeline
  from models.SAM.segment_anything import sam_model_registry, SamPredictor
  from models.AdaIN.inference import StyleTransfer

  progress_bar = st.progress(0, text="0/3")
  st.session_state.stable_diffusion = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16)
  progress_bar.progress(33, text="1/3")
  st.session_state.sam = sam_model_registry["vit_h"](checkpoint="models/SAM/segment_anything/model_zoo/sam_vit_h.pth")
  progress_bar.progress(66, text="2/3")
  st.session_state.adain = StyleTransfer(enc_path="models/AdaIN/model_zoo/encoder.pth", dec_path="models/AdaIN/model_zoo/decoder.pth")
  progress_bar.progress(100, text="3/3")

  st.session_state.stable_diffusion.to("cuda")
  st.session_state.sam.to("cuda")
  st.session_state.adain.to("cuda")
  st.session_state.sam_manual = SamPredictor(st.session_state.sam)

  st.session_state.models = True
  empty_memory()
  st.experimental_rerun()

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
if "no_image" not in st.session_state:
  st.session_state.no_image = Image.open("image/style/no_apply.png")
if "new_image" not in st.session_state:
  st.session_state.new_image = None
if "merged_mask" not in st.session_state:
  st.session_state.merged_mask = None
if "pairs" not in st.session_state:
  st.session_state.pairs = []

############################################
###### Step1 : Content Image
############################################

if not(st.session_state.step1):
  st.subheader("Step1 : Content Image")
  with st.expander("See Explanation", expanded=False):
    st.markdown("In this step, load or create an image of which style will be changed, which is called **'content image'**.")
    st.markdown("If you are satisfied with the content image, you can click **'next'** button to go to the next step.")

  st.divider()

  ## select how to load image
  option = st.radio("Select how to load the content image",("upload","url","path","create"))

  ## load image from upload
  if option == "upload":
    file = st.file_uploader("Upload content image", type=["png","jpg","jpeg"])
    if file is not None:
      image = Image.open(file).convert("RGB")
    else:
      image = None

  ## load image from url
  elif option == "url":
    with st.form(key="url"):
      url = st.text_input("Enter the url")
      button_url = st.form_submit_button("load")
      if button_url:
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
      else:
        image = None

  ## load image from path
  elif option == "path":
    with st.form(key="path"):
      path = st.text_input("Enter the path")
      button_path = st.form_submit_button("load")
      if button_path:
        image = Image.open(path).convert("RGB")
      else:
        image = None

  ## create image
  elif option == "create":
    guidance_scale_help = "Guidance scale controls how similar the created image will be to the text. \
    With higher guidance scale, text-to-image model will create an image that follows the text more strictly. \
    In contrast, with lower guidance scale, text-to-image model will create an image more creatively."

    diffusion_step_help = "Diffusion step means the number of image denoising steps. \
    If setting higher diffusion step, text-to-image model will created an image of higher quality with a long time. \
    We set 25 steps as the default value because an created image during 25 steps has sufficiently high quality in a reasonable time."

    with st.form(key="create"):
      text = st.text_input("Enter the text description of the image which you want to create")
      guidance_scale = st.slider("Select guidance scale value", 2.5, 7.5, 5.0, help=guidance_scale_help)
      diffusion_step = st.slider("Select diffusion step value", 1, 50, 25, help=diffusion_step_help)
      button_create = st.form_submit_button("create")
      if button_create:
        image = st.session_state.stable_diffusion(text, guidance_scale=guidance_scale, num_inference_steps=diffusion_step).images[0]
      else:
        image = None

  if image is not None:
    ## resize image for memory when either width or height is larger than 512
    w, h = image.size
    if max(w,h) > 512:
      resize_scale = 512 / max(w,h)
      image = image.resize((int(w*resize_scale), int(h*resize_scale)))

    st.session_state.content_image = image

  if st.session_state.content_image is not None:
    ## show content image
    st.markdown("##### Showing Content Image")
    st.image(st.session_state.content_image)

  st.divider()

  ## "next" button
  button_next = st.button("next", key="next")
  if button_next:
    if st.session_state.content_image is None:
      st.error("There's no content image!")
    else:
      st.session_state.step1 = True
      empty_memory()
      st.experimental_rerun()

  empty_memory()

############################################
###### Step2 : Mask & Style Image Pairs
############################################

if st.session_state.step1 and not(st.session_state.step2):
  st.subheader("Step2 : Mask & Style Image Pairs")
  with st.expander("See Explanation", expanded=False):
    st.markdown("In this step, choose mask from the content image and create pair it with the corresponding style image for each mask.")
    st.markdown("In here, **'style image'** means the image which has the artistic style or visual characteristics that we'd like to transfer to the content image.")
    st.markdown("If you want to create new content image, you can click **'prev'** button to go back to the previous step.")
    st.markdown("If you are satisfied with the created mask & style image pairs, you can click **'next'** button to go to the next step.")

  st.divider()

  if st.session_state.get_mask:
    tab1, tab2 = st.tabs(["Create Pairs", "Show Pairs"])

    ## create mask & style image pairs (mask)
    with tab1:
      ## select mask
      st.session_state.sam_manual.set_image(np.array(st.session_state.content_image))
      st.markdown("##### &nbsp; Click the part of which you want to get mask")
      coordinates = streamlit_image_coordinates(st.session_state.content_image, key='pil')

      with st.form(key="select_mask"):
        col1, col2 = st.columns([1,1])

        ## show selected mask
        with col1:
          st.markdown("##### Showing Selected Mask")

          if coordinates is not None:
            coordinates = np.array([[coordinates["x"],coordinates["y"]]])
            selected_mask, _, _ = st.session_state.sam_manual.predict(point_coords=coordinates, point_labels=np.array([1]), multimask_output=False)
            selected_mask = np.squeeze(selected_mask, axis=0)
            new_image = masked_image(st.session_state.content_image, selected_mask, point=coordinates[0])
            st.image(new_image, width=300)
          else:
            selected_mask = None
            st.image(st.session_state.content_image, width=300)

          ## "add to merged mask" button
          button_add = st.form_submit_button("add to Merged Mask",)
          if button_add:
            if selected_mask is None:
              st.error("There's no selected mask!")
            else:
              if st.session_state.merged_mask is None:
                st.session_state.merged_mask = selected_mask
              else:
                st.session_state.merged_mask = merge_masks([st.session_state.merged_mask, selected_mask])

        ## show merged mask
        with col2:
          st.markdown("##### Showing Merged Mask")

          if st.session_state.merged_mask is not None:
            st.image(masked_image(st.session_state.content_image, st.session_state.merged_mask), width=300)
          else:
            st.image(st.session_state.content_image, width=300)

          ## "reset mask", "go to select style image", "no apply style image" button
          button_reset_mask = st.form_submit_button("reset mask")
          button_style = st.form_submit_button("go to select style image")
          button_noapply1 = st.form_submit_button("no apply style image")

          if button_reset_mask:
            st.session_state.merged_mask = None
            empty_memory()
            st.experimental_rerun()

          if button_style:
            if st.session_state.merged_mask is None:
              st.error("There's no merged mask!")
            else:
              st.session_state.get_mask = False
              st.session_state.style_image = None
              empty_memory()
              st.experimental_rerun()

          if button_noapply1:
            if st.session_state.merged_mask is None:
              st.error("There's no merged mask!")
            else:
              st.session_state.pairs.append([st.session_state.merged_mask, None])
              st.session_state.merged_mask = None
              empty_memory()
              st.experimental_rerun()

      if len(st.session_state.pairs) == 0:
        mask_all = (np.ones_like(np.array(st.session_state.content_image)[:,:,0], dtype=bool) == False)
      else:
        mask_all = merge_masks([pair[0] for pair in st.session_state.pairs])

      ## select the mask not included mask & style image pairs
      if mask_all.sum() != mask_all.shape[0]*mask_all.shape[1]:
        with st.form(key="select2"):
          st.markdown("##### Showing The Mask Not Included Mask & Style Image Pairs")
          col1, col2 = st.columns([1,1])

          ## show the mask not included mask & style image pairs
          with col1:
            if len(st.session_state.pairs) == 0:
              mask_bg = np.ones_like(np.array(st.session_state.content_image)[:,:,0], dtype=bool)
            else:
              mask_bg = (merge_masks([pair[0] for pair in st.session_state.pairs]) == False)

            st.image(masked_image(st.session_state.content_image, mask_bg), width=300)

          ## "go to style image", "no apply style image" button
          with col2:
            st.markdown("")
            button_background = st.form_submit_button("go to select style image")
            button_noapply2 = st.form_submit_button("no apply style image")
            st.warning("If you do not click on any of the two buttons above before go to the next step, \
            the mask part on the left is fixed in black when style transfered.", icon="⚠️")

            if button_background:
              st.session_state.merged_mask = mask_bg
              st.session_state.get_mask = False
              st.session_state.style_image = None
              empty_memory()
              st.experimental_rerun()

            if button_noapply2:
              st.session_state.pairs.append([mask_bg, None])
              empty_memory()
              st.experimental_rerun()

      st.divider()

      ## "prev", "next" button
      col1, col2 = st.columns([1,10])
      with col1:
        button_prev = st.button("prev", key="prev")
      with col2:
        button_next = st.button("next", key="next")

      if button_prev:
        st.session_state.step1 = False
        st.session_state.style_image = None
        st.session_state.merged_mask = None
        st.session_state.pairs = []
        empty_memory()
        st.experimental_rerun()

      if button_next:
        if len(st.session_state.pairs) == 0:
          st.error("To continue next step, you should make 1 more mask & image pair!")
        else:
          st.session_state.step2 = True
          empty_memory()
          st.experimental_rerun()

      empty_memory()

    ## show created mask & style image pairs
    with tab2:
      st.markdown("##### Showing Mask & Style Image Pairs")

      if len(st.session_state.pairs) != 0:
        for idx, (mask, style_image) in enumerate(st.session_state.pairs):
          with st.form(key=f"pair {idx+1}"):
            st.markdown(f"Pair {idx+1}")
            if style_image is None:
              style_image = st.session_state.no_image
            col1, col2 = st.columns([1,1])
            with col1:
              st.image(masked_image(st.session_state.content_image, mask), width=300)
            with col2:
              st.image(style_image, width=300)
            button_reset = st.form_submit_button("reset pair")
            if button_reset:
              del st.session_state.pairs[idx]
              empty_memory()
              st.experimental_rerun()
      else:
        st.markdown("There's no mask & style image pair!")

      st.divider()

      ## "reset all pairs" button
      button_reset_all = st.button("reset all pairs", key="reset_pairs")
      if button_reset_all:
        st.session_state.pairs = []
        empty_memory()
        st.experimental_rerun()

  ## create mask & style image pairs (style image)
  if not(st.session_state.get_mask):
    ## show current mask
    st.markdown("##### Showing Current Mask")
    st.image(masked_image(st.session_state.content_image, st.session_state.merged_mask), width=300)

    ## select how to load image
    option = st.radio("Select how to load the style image",("upload","url","create","sample image"))

    ## load image from upload
    if option == "upload":
      file = st.file_uploader("Upload content image", type=["png","jpg","jpeg"])
      if file is not None:
        image = Image.open(file).convert("RGB")
      else:
        image = None

    ## load image from url
    elif option == "url":
      with st.form(key="url"):
        url = st.text_input("Enter the url")
        button_url = st.form_submit_button("load")
        if button_url:
          image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        else:
          image = None

    ## create image
    elif option == "create":
      guidance_scale_help = "Guidance scale controls how similar the created image will be to the text. \
      With higher guidance scale, text-to-image model will create an image that follows the text more strictly. \
      In contrast, with lower guidance scale, text-to-image model will create an image more creatively."

      diffusion_step_help = "Diffusion step means the number of image denoising steps. \
      If setting higher diffusion step, text-to-image model will created an image of higher quality with a long time. \
      We set 25 steps as the default value because an created image during 25 steps has sufficiently high quality in a reasonable time."

      with st.form(key="create"):
        text = st.text_input("Enter the text description of the image which you want to create")
        guidance_scale = st.slider("Select guidance scale value", 2.5, 7.5, 5.0, help=guidance_scale_help)
        diffusion_step = st.slider("Select diffusion step value", 1, 50, 25, help=diffusion_step_help)
        button_create = st.form_submit_button("create")
        if button_create:
          image =  st.session_state.stable_diffusion(text, guidance_scale=guidance_scale, num_inference_steps=diffusion_step).images[0]
        else:
          image = None

    ## sample image
    elif option == "sample image":
      sample_dict = {"select": None, 
                    "antimonocromatismo": "image/style/antimonocromatismo.jpg", 
                    "asheville": "image/style/asheville.jpg", 
                    "picasso seated nude hr": "image/style/picasso_seated_nude_hr.jpg", 
                    "brushstrokes": "image/style/brushstrokes.jpg",
                    "picasso self portrait": "image/style/picasso_self_portrait.jpg", 
                    "contrast of forms": "image/style/contrast_of_forms.jpg", 
                    "scene de rue": "image/style/scene_de_rue.jpg", 
                    "en campo gris": "image/style/en_campo_gris.jpg",
                    "sketch elsa": "image/style/sketch_elsa.jpeg", 
                    "flower of life": "image/style/flower_of_life.jpg", 
                    "the resevoir at poitiers": "image/style/the_resevoir_at_poitiers.jpg", 
                    "trial": "image/style/trial.jpg", 
                    "la muse": "image/style/la_muse.jpg",
                    "mondrian": "image/style/mondrian.jpg", 
                    "woman with hat matisse": "image/style/woman_with_hat_matisse.jpg", 
                    "starry night": "image/style/van_gogh_starry_night.jpeg"}
      style = st.selectbox("Select style image which you want", list(sample_dict.keys()))
      if style != "select":
        image = Image.open(sample_dict[style]).convert("RGB")
      else:
        image = None

    if image is not None:
      ## resize image for memory when either width or height is larger than 512
      w, h = image.size
      if max(w,h) > 512:
        resize_scale = 512 / max(w,h)
        image = image.resize((int(w*resize_scale), int(h*resize_scale)))

      st.session_state.style_image = image

    if st.session_state.style_image is not None:
      ## show style image
      st.markdown("##### Showing Style Image")
      st.image(st.session_state.style_image)

    st.divider()

    ## "select image", "not select image" button
    col1, col2 = st.columns([3,14])
    with col1:
      button_select = st.button("select image", key="select")
    with col2:
      button_not_select = st.button("not select image", key="not_select")

    if button_select:
      if st.session_state.style_image is None:
        st.error("There's no selected image!")
      else:
        st.session_state.get_mask = True
        st.session_state.pairs.append([st.session_state.merged_mask, st.session_state.style_image])
        st.session_state.merged_mask = None
        st.session_state.style_image = None
        empty_memory()
        st.experimental_rerun()

    if button_not_select:
      st.session_state.get_mask = True
      st.session_state.merged_mask = None
      st.session_state.style_image = None
      empty_memory()
      st.experimental_rerun()

    empty_memory()

############################################
###### Step3 : Style Transfer
############################################

if st.session_state.step1 and st.session_state.step2:
  st.subheader("Step3 : Style Transfer")
  with st.expander("See Explanation", expanded=False):
    st.markdown("In this step, apply the style of the style image to the mask part of the content image for all mask & style image pairs.")
    st.markdown("If you want to create new mask & style image pairs, you can click **'prev'** button to go back the previous step.")

  st.divider()

  ## show created mask & style image pairs
  st.markdown("##### Showing Created Mask & Style Image Pairs")
  for idx, (mask, style_image) in enumerate(st.session_state.pairs):
    if style_image is None:
      style_image = st.session_state.no_image
    st.markdown(f"Pair {idx+1}")
    col1, col2 = st.columns([1,1])
    with col1:
      st.image(masked_image(st.session_state.content_image, mask), width=300)
    with col2:
      st.image(style_image, width=300)

  ## style transfer
  alpha_help = "Alpha determines how much style of the style image will be applied to the content image. \
  With higher alpha, style transfer model will concentrate on the style of the style image. \
  In contraset, with lower alpha, style transfer model will concentrate on the style of the content image."

  preserve_color_help = "If checked, the overall color of the content image will be preserved when style transferd. \
  However, if not checked, the overall color of the style image will be applied to style transferd image."

  with st.form("style_transfer"):
    option_list = []
    for idx, (mask, style_image) in enumerate(st.session_state.pairs):
      if style_image is not None:
        alpha = st.slider(f"Select alpha value (Pair {idx+1})", 0.0, 1.0, 0.5, key=f"alpha_{idx+1}", help=alpha_help)
        preserve_color = st.checkbox(f"Preserve color (Pair {idx+1})", True, key=f"preserve_color_{idx+1}", help=preserve_color_help)
        option_list.append([alpha, preserve_color])
      else:
        option_list.append([None, None])
    button_run = st.form_submit_button("style transfer")

    if button_run:
      new_image = np.zeros_like(np.array(st.session_state.content_image), dtype=np.float32)

      ## create style transfered image
      for (mask, style_image), (alpha, preserve_color) in zip(st.session_state.pairs, option_list):
        ## not style transfer for the mask part
        if style_image is None:
          new_image += (np.array(st.session_state.content_image)/255) * np.stack([mask,mask,mask], axis=2)
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
    ## show style transferd image, "download" button
    st.markdown("##### Showing Style Transferd Image")
    st.image(st.session_state.new_image)
    buf = BytesIO()
    st.session_state.new_image.save(buf, format="png")
    st.download_button(label="download", data=buf.getvalue(), file_name="new_image.png", mime="image/png")

  st.divider()

  ## "prev" button
  button_prev = st.button("prev", key="prev")
  if button_prev:
    st.session_state.step2 = False
    st.session_state.new_image = None
    empty_memory()
    st.experimental_rerun()

  empty_memory()
