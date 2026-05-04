"""Inference-only SAMRoad model.

Slimmed from upstream `htcr/sam_road/model.py`. Drops:
- pytorch-lightning base (uses bare `nn.Module`)
- LoRA, NO_SAM ablation, USE_SAM_DECODER variant, FOCAL_LOSS
- training_step/validation_step/test_step/configure_optimizers
- torchmetrics, wandb, matplotlib, torchvision dependencies

Only `infer_masks_and_img_features` and `infer_toponet` (the two methods
used by the inference loop) are preserved.
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from segment_anything.modeling.image_encoder import ImageEncoderViT
from segment_anything.modeling.common import LayerNorm2d


_SAM_VARIANTS = {
    "vit_b": dict(embed_dim=768, depth=12, num_heads=12, global_attn_indexes=[2, 5, 8, 11]),
    "vit_l": dict(embed_dim=1024, depth=24, num_heads=16, global_attn_indexes=[5, 11, 17, 23]),
    "vit_h": dict(embed_dim=1280, depth=32, num_heads=16, global_attn_indexes=[7, 15, 23, 31]),
}


class BilinearSampler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, feature_maps, sample_points):
        # feature_maps: [B, D, H, W]; sample_points: [B, N, 2] in pixel coords.
        sample_points = (sample_points / self.config.PATCH_SIZE) * 2.0 - 1.0
        sample_points = sample_points.unsqueeze(2)
        sampled = F.grid_sample(
            feature_maps, sample_points, mode="bilinear", align_corners=False
        )
        return sampled.squeeze(dim=-1).permute(0, 2, 1)


class TopoNet(nn.Module):
    def __init__(self, config, feature_dim):
        super().__init__()
        self.config = config
        self.hidden_dim = 128
        self.heads = 4
        self.num_attn_layers = 3

        self.feature_proj = nn.Linear(feature_dim, self.hidden_dim)
        self.pair_proj = nn.Linear(2 * self.hidden_dim + 2, self.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.heads,
            dim_feedforward=self.hidden_dim,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        if self.config.TOPONET_VERSION != "no_transformer":
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=self.num_attn_layers
            )
        self.output_proj = nn.Linear(self.hidden_dim, 1)

    def forward(self, points, point_features, pairs, pairs_valid):
        point_features = F.relu(self.feature_proj(point_features))
        batch_size, n_samples, n_pairs, _ = pairs.shape
        pairs = pairs.view(batch_size, -1, 2)

        batch_indices = torch.arange(batch_size, device=pairs.device).view(-1, 1).expand(
            -1, n_samples * n_pairs
        )
        src_features = point_features[batch_indices, pairs[:, :, 0]]
        tgt_features = point_features[batch_indices, pairs[:, :, 1]]
        src_points = points[batch_indices, pairs[:, :, 0]]
        tgt_points = points[batch_indices, pairs[:, :, 1]]
        offset = tgt_points - src_points

        if self.config.TOPONET_VERSION == "no_tgt_features":
            pair_features = torch.cat(
                [src_features, torch.zeros_like(tgt_features), offset], dim=2
            )
        elif self.config.TOPONET_VERSION == "no_offset":
            pair_features = torch.cat(
                [src_features, tgt_features, torch.zeros_like(offset)], dim=2
            )
        else:
            pair_features = torch.cat([src_features, tgt_features, offset], dim=2)

        pair_features = F.relu(self.pair_proj(pair_features))
        pair_features = pair_features.view(batch_size * n_samples, n_pairs, -1)
        pairs_valid = pairs_valid.view(batch_size * n_samples, n_pairs)

        # Flip the mask on rows that are entirely invalid — otherwise the
        # transformer's softmax over an all-padding row produces NaN.
        all_invalid = torch.eq(torch.sum(pairs_valid, dim=-1), 0).unsqueeze(-1)
        pairs_valid = torch.logical_or(pairs_valid, all_invalid)
        padding_mask = ~pairs_valid

        if self.config.TOPONET_VERSION != "no_transformer":
            pair_features = self.transformer_encoder(
                pair_features, src_key_padding_mask=padding_mask
            )

        _, n_pairs, _ = pair_features.shape
        pair_features = pair_features.view(batch_size, n_samples, n_pairs, -1)

        logits = self.output_proj(pair_features)
        scores = torch.sigmoid(logits)
        return logits, scores


class SAMRoad(nn.Module):
    """Inference-only SAMRoad: SAM ViT image encoder + naive map decoder + TopoNet."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        sam_version = config.SAM_VERSION
        if sam_version not in _SAM_VARIANTS:
            raise ValueError(f"unsupported SAM_VERSION: {sam_version!r}")
        v = _SAM_VARIANTS[sam_version]

        prompt_embed_dim = 256
        image_size = config.PATCH_SIZE
        self.image_size = image_size
        vit_patch_size = 16
        encoder_output_dim = prompt_embed_dim

        self.register_buffer(
            "pixel_mean",
            torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
            persistent=False,
        )

        self.image_encoder = ImageEncoderViT(
            depth=v["depth"],
            embed_dim=v["embed_dim"],
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=v["num_heads"],
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=v["global_attn_indexes"],
            window_size=14,
            out_chans=prompt_embed_dim,
        )

        # Naive 4×-upsample decoder (output channels: keypoint + road).
        activation = nn.GELU
        self.map_decoder = nn.Sequential(
            nn.ConvTranspose2d(encoder_output_dim, 128, kernel_size=2, stride=2),
            LayerNorm2d(128),
            activation(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            activation(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            activation(),
            nn.ConvTranspose2d(32, 2, kernel_size=2, stride=2),
        )

        self.bilinear_sampler = BilinearSampler(config)
        self.topo_net = TopoNet(config, encoder_output_dim)

    @torch.inference_mode()
    def infer_masks_and_img_features(self, rgb):
        # rgb: [B, H, W, C] uint-ish float32
        x = rgb.permute(0, 3, 1, 2)
        x = (x - self.pixel_mean) / self.pixel_std
        image_embeddings = self.image_encoder(x)
        mask_logits = self.map_decoder(image_embeddings)
        mask_scores = torch.sigmoid(mask_logits)
        mask_scores = mask_scores.permute(0, 2, 3, 1)  # [B, H, W, 2]
        return mask_scores, image_embeddings

    @torch.inference_mode()
    def infer_toponet(self, image_embeddings, graph_points, pairs, valid):
        point_features = self.bilinear_sampler(image_embeddings, graph_points)
        _, topo_scores = self.topo_net(graph_points, point_features, pairs, valid)
        return topo_scores
