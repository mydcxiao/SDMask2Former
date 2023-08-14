import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .position_encoding import PositionEmbeddingSine
from .mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder, MLP

from timm.models.layers import DropPath, drop_path

class VLFusion(nn.Module):
    def __init__(
        self,
        v_channels: int,
        l_channels: int,
        hidden_dim: int,
        nheads: int = 8,
        init_values: float = 1e-4,
        dropout: float = 0.1,
        droppath: float = 0.4,
    ):
        super().__init__()
        self.v_channels = v_channels
        self.l_channels = l_channels
        self.hidden_dim = hidden_dim
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, nheads, dropout=dropout)
        self.droppath = droppath
        self.v_proj = nn.Linear(v_channels, hidden_dim)
        self.l_proj = nn.Linear(l_channels, hidden_dim)
        self.out_v_proj = nn.Linear(hidden_dim, v_channels)
        self.v_norm = nn.LayerNorm(v_channels)

        self.gamma_v = nn.Parameter(init_values * torch.ones((v_channels)), requires_grad=True)
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, vis, l, l_masks=None):
        b, c, h, w = vis.shape
        vis = vis.flatten(2).transpose(1, 2) # b x hw x c
        q = self.v_proj(vis)
        k = v = self.l_proj(l)
        if l_masks is not None:
            q = self.multihead_attn(query=q.transpose(0, 1),
                                    key=k.transpose(0, 1),
                                    value=v.transpose(0, 1),
                                    key_padding_mask=l_masks)[0]
        else:
            q = self.multihead_attn(query=q.transpose(0, 1),
                                    key=k.transpose(0, 1),
                                    value=v.transpose(0, 1))[0]
        out_v = self.out_v_proj(q.transpose(0, 1))
        vis = vis + drop_path(self.gamma_v * out_v, self.droppath, self.training)
        vis = self.v_norm(vis)
        vis = vis.transpose(1, 2).reshape(b, c, h, w)
        return vis

class QueryLoad(nn.Module):
    pass


class RefMultiScaleMaskedTransformerDecoder(MultiScaleMaskedTransformerDecoder):
    def __init__(
        self,
        *,
        l_channels: int = 768,
        init_values: float = 1e-4,
        dropout: float = 0.0,
        droppath: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert self.mask_classification

        self.vlfusion = VLFusion(
            v_channels=kwargs['hidden_dim'],
            l_channels=l_channels,
            hidden_dim=kwargs['hidden_dim'],
            nheads=kwargs['nheads'],
            init_values=init_values,
            dropout=dropout,
            droppath=droppath,
        )

        self.sim_embed = MLP(kwargs['hidden_dim'], kwargs['hidden_dim'], l_channels, 3)
    
    def forward(self, x, mask_features, l_embed, l_encodings, l_masks=None, mask = None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_sim = []
        predictions_mask = []

        # vlfusion for mask_features
        mask_features = self.vlfusion(mask_features, l_encodings, l_masks=l_masks)

        # prediction heads on learnable query features
        outputs_sim, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, l_embed, attn_mask_target_size=size_list[0])
        predictions_sim.append(outputs_sim)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_sim, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, l_embed, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_sim.append(outputs_sim)
            predictions_mask.append(outputs_mask)

        assert len(predictions_sim) == self.num_layers + 1

        out = {
            'pred_logits': predictions_sim[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_sim if self.mask_classification else None, predictions_mask
            )
        }
        return out

    def forward_prediction_heads(self, output, mask_features, l_embed, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        # outputs_class = self.class_embed(decoder_output)
        sim_embed = self.sim_embed(decoder_output)
        outputs_sim = torch.einsum("bqc,bc->bq", sim_embed, l_embed)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_sim, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
