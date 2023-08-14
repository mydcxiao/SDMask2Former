import torch
import torch.nn as nn

from typing import Any, Dict, List, Tuple, Optional

from .ldm_encoder.meta_arch.ldm import LdmImplicitCaptionerExtractor
from .ldm_encoder.backbone.feature_extractor import FeatureExtractorBackbone
from .pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from .transformer_decoder import RefMultiScaleMaskedTransformerDecoder

class M(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: FeatureExtractorBackbone,
        pixel_decoder: MSDeformAttnPixelDecoder,
        mask_decoder: RefMultiScaleMaskedTransformerDecoder,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.pixel_decoder = pixel_decoder
        self.mask_decoder = mask_decoder
        
        self._freeze_image_encoder()
        # self._freeze_pixel_decoder()

    def forward(
        self,
        batched_images, # B x C x H x W
        batched_sents, # B x 1 x 77
    ) -> List[Dict[str, torch.Tensor]]:
        image_embeddings = self.image_encoder(batched_images)
        mask_features, _, multi_scale_features = self.pixel_decoder.forward_features(image_embeddings)
        clip = self.image_encoder.feature_extractor.clip
        batched_sents = batched_sents.squeeze(1)
        l_masks = (batched_sents == 0).bool()
        l_embed, l_encodings = clip._encode_text(batched_sents)
        l_embed, l_encodings = l_embed.float(), l_encodings.float()
        out = self.mask_decoder(multi_scale_features, mask_features, l_embed, l_encodings, l_masks=l_masks)
        return out
    
    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_image_encoder()
        # self._freeze_pixel_decoder()
        return self

    def _freeze_image_encoder(self):
        self.image_encoder.eval()
        for p in self.image_encoder.parameters():
            p.requires_grad = False

    def _freeze_pixel_decoder(self):
        self.pixel_decoder.eval()
        for p in self.pixel_decoder.parameters():
            p.requires_grad = False
