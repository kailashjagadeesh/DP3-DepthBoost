import torch
import torch.nn as nn
import torchvision.transforms as T
from termcolor import cprint
from typing import Dict

# Reuse the create_mlp from pointnet_extractor
from diffusion_policy_3d.model.vision.pointnet_extractor import create_mlp

class DinoV3Encoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 img_crop_shape=None, # Expecting [512, 512]
                 state_mlp_size=(64, 64), 
                 state_mlp_activation_fn=nn.ReLU,
                 dino_model_name='dinov3_vitl16', # Note: 'dinov3' usually uses patch size 16
                 freeze_backbone=True,
                 **kwargs
                 ):
        super().__init__()
        
        self.image_key = 'image' 
        self.state_key = 'agent_pos'
        
        cprint(f"[DinoV3Encoder] Loading {dino_model_name} from facebookresearch/dinov3...", "cyan")
        
        # Load DINOv3 from official Hub
        # trust_repo=True is often required for new private/public repos with custom code
        try:
            self.backbone = torch.hub.load('facebookresearch/dinov3', dino_model_name, trust_repo=True)
        except Exception as e:
            cprint(f"Standard Hub Load failed: {e}. Trying local fallback or Transformers...", "red")
            # Fallback: You might need to install via transformers if Hub fails
            # from transformers import AutoModel
            # self.backbone = AutoModel.from_pretrained(f"facebook/{dino_model_name}")
            raise e

        # Determine embedding dimension (e.g., 1024 for ViT-L)
        self.visual_feature_dim = self.backbone.embed_dim
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        
        # ImageNet Normalization (Required for DINO)
        self.normalize = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # State MLP
        self.state_shape = observation_space[self.state_key]
        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        state_output_dim = state_mlp_size[-1]

        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], state_output_dim, net_arch, state_mlp_activation_fn))

        self.n_output_channels = self.visual_feature_dim + state_output_dim
        
        cprint(f"[DinoV3Encoder] Visual Dim: {self.visual_feature_dim} | State Dim: {state_output_dim}", "yellow")

    def forward(self, observations: Dict) -> torch.Tensor:
        # Input: [B, T, C, H, W] or [B, C, H, W]
        images = observations[self.image_key]
        
        # Flatten Time Dimension
        has_time_dim = len(images.shape) == 5
        if has_time_dim:
            B, T_steps, C, H, W = images.shape
            images = images.view(B * T_steps, C, H, W)
        
        # Ensure 0-1 range before normalization
        if images.max() > 1.0:
            images = images / 255.0
            
        # DINOv3 Forward Pass
        processed_imgs = self.normalize(images)
        
        # DINOv3 often returns a dictionary or specific tensor. 
        # Using forward_features() is safer to get the CLS token or patch embeddings.
        # We usually want the CLS token (global descriptor) for diffusion conditioning.
        features_dict = self.backbone.forward_features(processed_imgs)
        visual_feat = features_dict['x_norm_clstoken'] # [B*T, Embed_Dim]

        # Process State
        state = observations[self.state_key]
        if has_time_dim:
            state = state.view(B * T_steps, -1)
            
        state_feat = self.state_mlp(state)
        
        # Concatenate
        final_feat = torch.cat([visual_feat, state_feat], dim=-1)
        
        if has_time_dim:
            final_feat = final_feat.view(B, T_steps, -1)
            
        return final_feat

    def output_shape(self):
        return self.n_output_channels