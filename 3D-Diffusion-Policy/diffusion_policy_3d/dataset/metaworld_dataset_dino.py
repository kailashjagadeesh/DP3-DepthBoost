"""
MetaWorld Dataset for DINOv2-based DP3
Loads RGB images instead of point clouds for vision-based policy learning.
"""

from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset


class MetaworldDatasetDino(BaseDataset):
    """
    MetaWorld dataset that loads RGB images for DINOv2 encoder.
    
    This dataset loads:
    - 'img': RGB images for visual encoding
    - 'state': Robot state (agent_pos)
    - 'action': Robot actions
    
    Images are stored as uint8 [0-255] in zarr and converted to float32 [0-1] here.
    """
    
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            image_size=128,  # Expected image size (used for validation)
            ):
        super().__init__()
        
        # Load data from zarr - now including 'img'
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action', 'img'])
        
        self.image_size = image_size
        
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        """
        Get normalizer for actions and agent positions.
        Images are normalized separately in the encoder (ImageNet normalization).
        """
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:],
            # Note: We don't normalize images here - DINOv2 encoder handles that
            # with ImageNet normalization
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        
        # Add a pass-through normalizer for images (no normalization)
        # This is needed because the normalizer expects all observation keys
        image_shape = self.replay_buffer['img'].shape[1:]  # (H, W, C) or (C, H, W)
        normalizer['image'] = SingleFieldLinearNormalizer.create_identity()
        
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        """Convert a sample from replay buffer to training data format."""
        agent_pos = sample['state'][:, :].astype(np.float32)
        
        # Process images
        # Images from zarr are typically (T, H, W, C) uint8
        # We need (T, C, H, W) float32 for PyTorch
        images = sample['img'].astype(np.float32)
        
        # If images are channel-last (H, W, C), convert to channel-first (C, H, W)
        if images.shape[-1] == 3:  # (T, H, W, C) -> (T, C, H, W)
            images = np.transpose(images, (0, 3, 1, 2))
        
        # Keep images in [0, 255] range - the encoder will normalize them
        # (DINOv2 encoder divides by 255 if max > 1.0)

        data = {
            'obs': {
                'image': images,
                'agent_pos': agent_pos, 
            },
            'action': sample['action'].astype(np.float32)
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data



