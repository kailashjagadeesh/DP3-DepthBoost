"""
DINOv2 Vision Encoder for DP3
Uses HuggingFace Transformers to load DINOv2 models.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
from termcolor import cprint
from typing import Dict

# Reuse the create_mlp from pointnet_extractor
from diffusion_policy_3d.model.vision.pointnet_extractor import create_mlp

# HuggingFace model name mapping
HF_MODEL_MAP = {
    # Short names -> HuggingFace names
    'dinov2_vits14': 'facebook/dinov2-small',
    'dinov2_vitb14': 'facebook/dinov2-base',
    'dinov2_vitl14': 'facebook/dinov2-large',
    'dinov2_vitg14': 'facebook/dinov2-giant',
    # Also support direct HuggingFace names
    'facebook/dinov2-small': 'facebook/dinov2-small',
    'facebook/dinov2-base': 'facebook/dinov2-base',
    'facebook/dinov2-large': 'facebook/dinov2-large',
    'facebook/dinov2-giant': 'facebook/dinov2-giant',
}

# Embedding dimensions for each model
EMBED_DIMS = {
    'facebook/dinov2-small': 384,
    'facebook/dinov2-base': 768,
    'facebook/dinov2-large': 1024,
    'facebook/dinov2-giant': 1536,
}


class DinoV2Encoder(nn.Module):
    """
    DINOv2 Vision Encoder using HuggingFace Transformers.
    Extracts visual features from RGB images using pre-trained DINOv2 models.
    """
    
    def __init__(self, 
                 observation_space: Dict, 
                 img_crop_shape=None,
                 state_mlp_size=(64, 64), 
                 state_mlp_activation_fn=nn.ReLU,
                 dino_model_name='dinov2_vitl14',  # Default to DINOv2-Large
                 freeze_backbone=True,
                 **kwargs
                 ):
        """
        Initialize DINOv2 encoder.
        
        Args:
            observation_space: Dict with observation shapes (must include 'image' and 'agent_pos')
            img_crop_shape: Optional crop shape for images
            state_mlp_size: Tuple of hidden layer sizes for state MLP
            state_mlp_activation_fn: Activation function for state MLP
            dino_model_name: Name of DINOv2 model to use
            freeze_backbone: Whether to freeze DINOv2 weights
        """
        super().__init__()
        
        self.image_key = 'image' 
        self.state_key = 'agent_pos'
        
        # Map model name to HuggingFace name
        hf_model_name = HF_MODEL_MAP.get(dino_model_name, dino_model_name)
        cprint(f"[DinoV2Encoder] Loading {hf_model_name} from HuggingFace...", "cyan")
        
        # Load from HuggingFace Transformers
        try:
            from transformers import AutoModel
            self.backbone = AutoModel.from_pretrained(hf_model_name)
            cprint(f"[DinoV2Encoder] Successfully loaded {hf_model_name}!", "green")
        except Exception as e:
            cprint(f"[DinoV2Encoder] HuggingFace load failed: {e}", "red")
            raise e

        # Get embedding dimension from model config or lookup table
        if hasattr(self.backbone.config, 'hidden_size'):
            self.visual_feature_dim = self.backbone.config.hidden_size
        else:
            self.visual_feature_dim = EMBED_DIMS.get(hf_model_name, 1024)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
            cprint(f"[DinoV2Encoder] Backbone frozen (no gradient updates)", "cyan")
        
        # ImageNet Normalization (required for DINO models)
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        
        # State MLP (same as PointNet encoder)
        self.state_shape = observation_space[self.state_key]
        if len(state_mlp_size) == 0:
            raise RuntimeError("State MLP size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = list(state_mlp_size[:-1])
        state_output_dim = state_mlp_size[-1]

        self.state_mlp = nn.Sequential(
            *create_mlp(self.state_shape[0], state_output_dim, net_arch, state_mlp_activation_fn)
        )

        # Total output dimension = visual features + state features
        self.n_output_channels = self.visual_feature_dim + state_output_dim
        
        cprint(f"[DinoV2Encoder] Visual Dim: {self.visual_feature_dim} | "
               f"State Dim: {state_output_dim} | Total: {self.n_output_channels}", "yellow")

    def forward(self, observations: Dict) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            observations: Dict containing 'image' and 'agent_pos' keys
                - image: (B, C, H, W) or (B, H, W, C) tensor
                - agent_pos: (B, state_dim) tensor
        
        Returns:
            Combined feature tensor of shape (B, n_output_channels)
        """
        # Get images from observations
        images = observations[self.image_key]
        
        # Ensure correct format: (B, C, H, W)
        if len(images.shape) == 4 and images.shape[-1] == 3:
            # (B, H, W, C) -> (B, C, H, W)
            images = images.permute(0, 3, 1, 2)
        
        # Ensure images are float and in [0, 1] range
        images = images.float()
        if images.max() > 1.0:
            images = images / 255.0
        
        # Apply ImageNet normalization
        processed_imgs = self.normalize(images)
        
        # DINOv2 Forward Pass
        # Control gradient computation based on training mode and frozen state
        with torch.set_grad_enabled(
            self.training and any(p.requires_grad for p in self.backbone.parameters())
        ):
            outputs = self.backbone(processed_imgs)
            
            # Get CLS token (first token of last_hidden_state)
            # HuggingFace DINOv2 returns last_hidden_state of shape (B, num_patches+1, hidden_size)
            # The first token [0] is the CLS token
            if hasattr(outputs, 'last_hidden_state'):
                visual_feat = outputs.last_hidden_state[:, 0]  # CLS token: (B, hidden_size)
            elif hasattr(outputs, 'pooler_output'):
                visual_feat = outputs.pooler_output
            else:
                # Fallback: use first token from tuple output
                visual_feat = outputs[0][:, 0]

        # Process robot state through MLP
        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # (B, state_output_dim)
        
        # Concatenate visual and state features
        final_feat = torch.cat([visual_feat, state_feat], dim=-1)
            
        return final_feat

    def output_shape(self):
        """Return the total output feature dimension."""
        return self.n_output_channels


# Alias for backward compatibility
DinoV3Encoder = DinoV2Encoder
