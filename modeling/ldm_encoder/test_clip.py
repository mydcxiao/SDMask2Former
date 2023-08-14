import torch
import torch.nn as nn
from meta_arch.clip import MaskCLIP
import open_clip


clip = MaskCLIP()
image = torch.randn(8, 3, 128, 128)
mask = torch.randn(8, 100, 128, 128)

mask_embed = clip.get_mask_embed(image, mask)
print(mask_embed.shape) # (8, 100, 768)