import torch
import numpy as np

from .net import vgg, decoder
from .function import adaptive_instance_normalization, coral


class StyleTransfer:

  def __init__(self, enc_path, dec_path):

    super().__init__()

    vgg.load_state_dict(torch.load(enc_path))
    decoder.load_state_dict(torch.load(dec_path))

    self.enc = vgg[:31]
    self.dec = decoder

    self.enc.eval()
    self.dec.eval()

  def run(self, content_img, style_img, alpha=1.0, preserve_color=True, device="cpu"):

    content_img = torch.FloatTensor(content_img).permute(2,0,1)
    style_img = torch.FloatTensor(style_img).permute(2,0,1)

    if preserve_color:
      style_img = coral(style_img, content_img)

    content_img = content_img.unsqueeze(0).to(device)
    style_img = style_img.unsqueeze(0).to(device)

    with torch.no_grad():
      content_feat = self.enc(content_img)
      style_feat = self.enc(style_img)
      new_feat = adaptive_instance_normalization(content_feat, style_feat)
      new_feat = alpha*new_feat + (1-alpha)*content_feat
      new_img = self.dec(new_feat)

    new_img = new_img.squeeze(0).permute(1,2,0).cpu().numpy()
    new_img = np.clip(new_img, 0, 1)

    return new_img
