from typing import Dict, Optional

import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn


class ConcatEncoders(nn.Module):
    def __init__(
        self,
        proprio: DictConfig,
        vision_static: Optional[DictConfig] = None,
        vision_gripper: Optional[DictConfig] = None,
        depth_static: Optional[DictConfig] = None,
        depth_gripper: Optional[DictConfig] = None,
        tactile: Optional[DictConfig] = None,
        seg_static: Optional[DictConfig] = None,
        seg_gripper: Optional[DictConfig] = None,
    ):
        super().__init__()
        self._latent_size = 0
        if vision_static:
            self._latent_size += vision_static.visual_features
        if vision_gripper:
            self._latent_size += vision_gripper.visual_features
        if depth_gripper and vision_gripper.num_c < 4:
            vision_gripper.num_c += depth_gripper.num_c
        if depth_static and vision_static.num_c < 4:
            vision_static.num_c += depth_static.num_c
        if tactile:
            self._latent_size += tactile.visual_features
        if seg_static:
            self._latent_size += seg_static.visual_features
        if seg_gripper:
            self._latent_size += seg_gripper.visual_features

        self.vision_static_encoder = hydra.utils.instantiate(vision_static)
        self.vision_gripper_encoder = hydra.utils.instantiate(vision_gripper) if vision_gripper else None
        self.proprio_encoder = hydra.utils.instantiate(proprio)
        self.tactile_encoder = hydra.utils.instantiate(tactile)
        self.seg_static_encoder = hydra.utils.instantiate(seg_static) if seg_static else None
        self.seg_gripper_encoder = hydra.utils.instantiate(seg_gripper) if seg_gripper else None
        self._latent_size += self.proprio_encoder.out_features

    @property
    def latent_size(self):
        return self._latent_size

    def forward(
        self, imgs: Dict[str, torch.Tensor], depth_imgs: Dict[str, torch.Tensor], decomp_imgs: Dict[str, torch.Tensor], state_obs: torch.Tensor
    ) -> torch.Tensor:
        # for debugging, print all info about the inputs
        # print("imgs: ", imgs)
        # print("depth_imgs: ", depth_imgs)
        # print("state_obs: ", state_obs)
        
        imgs_static = imgs["rgb_static"] if "rgb_static" in imgs else None
        imgs_gripper = imgs["rgb_gripper"] if "rgb_gripper" in imgs else None
        imgs_tactile = imgs["rgb_tactile"] if "rgb_tactile" in imgs else None
        depth_static = depth_imgs["depth_static"] if "depth_static" in depth_imgs else None
        # depth_static = depth_imgs["seg_static"] if "seg_static" in depth_imgs else None
        depth_gripper = depth_imgs["depth_gripper"] if "depth_gripper" in depth_imgs else None
        seg_static = decomp_imgs["seg_static"] if "seg_static" in decomp_imgs else None
        seg_gripper = decomp_imgs["seg_gripper"] if "seg_gripper" in decomp_imgs else None
        
        encoded_imgs = None
        if imgs_static is not None and self.vision_static_encoder:
            b, s, c, h, w = imgs_static.shape
            imgs_static = imgs_static.reshape(-1, c, h, w)  # (batch_size * sequence_length, 3, 200, 200)
            if depth_static is not None:
                depth_static = torch.unsqueeze(depth_static, 2)
                depth_static = depth_static.reshape(-1, 1, h, w)  # (batch_size * sequence_length, 1, 200, 200)
                imgs_static = torch.cat([imgs_static, depth_static], dim=1)  # (batch_size * sequence_length, 3, 200, 200)
            # ------------ Vision Network ------------ #
            encoded_imgs = self.vision_static_encoder(imgs_static)  # (batch*seq_len, 64)
            encoded_imgs = encoded_imgs.reshape(b, s, -1)  # (batch, seq, 64)

        if imgs_gripper is not None:
            b, s, c, h, w = imgs_gripper.shape
            imgs_gripper = imgs_gripper.reshape(-1, c, h, w)  # (batch_size * sequence_length, 3, 84, 84)
            if depth_gripper is not None:
                depth_gripper = torch.unsqueeze(depth_gripper, 2)
                depth_gripper = depth_gripper.reshape(-1, 1, h, w)  # (batch_size * sequence_length, 1, 84, 84)
                imgs_gripper = torch.cat(
                    [imgs_gripper, depth_gripper], dim=1
                )  # (batch_size * sequence_length, 4, 84, 84)
            encoded_imgs_gripper = self.vision_gripper_encoder(imgs_gripper)  # (batch*seq_len, 64)
            encoded_imgs_gripper = encoded_imgs_gripper.reshape(b, s, -1)  # (batch, seq, 64)
            if encoded_imgs is not None:
                encoded_imgs = torch.cat([encoded_imgs, encoded_imgs_gripper], dim=-1)
            else:
                encoded_imgs = encoded_imgs_gripper

        if imgs_tactile is not None:
            b, s, c, h, w = imgs_tactile.shape
            imgs_tactile = imgs_tactile.reshape(-1, c, h, w)  # (batch_size * sequence_length, 3, 84, 84)
            encoded_tactile = self.tactile_encoder(imgs_tactile)
            encoded_tactile = encoded_tactile.reshape(b, s, -1)
            if encoded_imgs is not None:
                encoded_imgs = torch.cat([encoded_imgs, encoded_tactile], dim=-1)
            else:
                encoded_imgs = encoded_tactile
                
        if seg_static is not None:
            seg_static = torch.unsqueeze(seg_static, 2)
            b, s, c, h, w = seg_static.shape
            seg_static = seg_static.reshape(-1, c, h, w)
            encoded_seg_static = self.seg_static_encoder(seg_static)
            encoded_seg_static = encoded_seg_static.reshape(b, s, -1)
            if encoded_imgs is not None:
                encoded_imgs = torch.cat([encoded_imgs, encoded_seg_static], dim=-1)
            else:
                encoded_imgs = encoded_seg_static
                
        if seg_gripper is not None:
            seg_gripper = torch.unsqueeze(seg_gripper, 2)
            b, s, c, h, w = seg_gripper.shape
            seg_gripper = seg_gripper.reshape(-1, c, h, w)
            encoded_seg_gripper = self.seg_gripper_encoder(seg_gripper)
            encoded_seg_gripper = encoded_seg_gripper.reshape(b, s, -1)
            if encoded_imgs is not None:
                encoded_imgs = torch.cat([encoded_imgs, encoded_seg_gripper], dim=-1)
            else:
                encoded_imgs = encoded_seg_gripper

        state_obs_out = self.proprio_encoder(state_obs)
        perceptual_emb = torch.cat([encoded_imgs, state_obs_out], dim=-1)
        return perceptual_emb
