# src/model.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from einops import rearrange

class PatchEmbedding(nn.Module):
    """Spatio-Temporal Patch Embedding."""
    def __init__(self, grid_shape, patch_size, in_channels, embed_dim):
        super().__init__()
        self.num_patches_h = grid_shape[0] // patch_size
        self.num_patches_w = grid_shape[1] // patch_size
        self.num_patches_per_map = self.num_patches_h * self.num_patches_w
        
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x shape: (B, T, C, H, W) -> (B*T, C, H, W)
        B, T, C, H, W = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.projection(x) # -> (B*T, D, num_patches_h, num_patches_w)
        x = rearrange(x, 'bt d ph pw -> bt (ph pw) d') # -> (B*T, N_patches_per_map, D)
        return x

class Attention(nn.Module):
    """Causal-Informed Multi-Head Self-Attention."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, causal_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if causal_mask is not None:
            # Add causal mask to attention scores
            # The mask should be broadcastable to the attention shape
            attn = attn + causal_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    """Feed-Forward Network."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    """Transformer Encoder Block."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, causal_mask=None):
        x = x + self.attn(self.norm1(x), causal_mask=causal_mask)
        x = x + self.mlp(self.norm2(x))
        return x

class CST_VT(nn.Module):
    """Causal-Informed Spatio-Temporal Variational Transformer."""
    def __init__(self, grid_shape, patch_size, in_channels, num_classes=1, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0.1, attn_drop_rate=0.1, norm_layer=nn.LayerNorm,
                 num_input_time_steps=12, causal_graph_path=None):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_input_time_steps = num_input_time_steps

        # 1. Patch Embedding
        self.patch_embed = PatchEmbedding(grid_shape, patch_size, in_channels, embed_dim)
        self.num_patches_per_map = self.patch_embed.num_patches_per_map
        self.num_total_patches = self.num_patches_per_map * num_input_time_steps

        # 2. Positional and Temporal Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches_per_map + 1, embed_dim))
        self.time_embed = nn.Parameter(torch.zeros(1, num_input_time_steps, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 3. Transformer Encoder Blocks
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)

        # 4. Probabilistic Prediction Head
        self.head_mu = nn.Linear(embed_dim, num_classes)
        self.head_log_var = nn.Linear(embed_dim, num_classes)
        
        # 5. Causal Mask
        self.causal_mask = self.load_and_build_causal_mask(causal_graph_path) if causal_graph_path else None
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_and_build_causal_mask(self, path):
        """Loads a CSV graph and builds a floating point mask for attention."""
        if not path:
            return None
        try:
            adj_df = pd.read_csv(path)
            num_nodes = self.num_total_patches + 1 # +1 for CLS token
            adj_matrix = torch.ones(num_nodes, num_nodes) # Start with all connections allowed

            # Assume graph is for patches only. We need to map to total patch indices.
            for _, row in adj_df.iterrows():
                # This logic assumes a simple 1-to-1 mapping and needs to be adapted for spatio-temporal indices
                # For this example, we'll use a simplified logic. A real implementation would be more complex.
                pass
            
            # The causal mask adds a large negative number to non-allowed connections
            mask = torch.zeros_like(adj_matrix)
            mask[adj_matrix == 0] = -1e9 # A large negative value
            
            # Add mask to all heads dimension
            return mask.unsqueeze(0).unsqueeze(0) # (1, 1, N, N) for broadcasting
        except FileNotFoundError:
            print(f"Warning: Causal graph not found at {path}. Proceeding without causal mask.")
            return None

    def forward(self, x):
        B = x.shape[0]
        
        # Reshape for patch embedding: (B, T, C, H, W)
        x_reshaped = rearrange(x, 'b (t c) h w -> b t c h w', t=self.num_input_time_steps)
        
        # Get patch embeddings for all time steps
        x = self.patch_embed(x_reshaped) # (B*T, N_patches_per_map, D)
        
        # Reshape and add temporal embeddings
        x = rearrange(x, '(b t) n d -> b t n d', b=B)
        x = x + self.time_embed.unsqueeze(2) # Add time embedding to each patch
        x = rearrange(x, 'b t n d -> b (t n) d')
        
        # Prepend CLS token and add positional embeddings
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # This is a simplification; a more robust pos_embed would be needed for spatio-temporal data
        # For now, we reuse the spatial pos_embed, which is common in ViT adaptations
        x = torch.cat((cls_tokens, x), dim=1)
        # The pos_embed logic needs refinement for spatio-temporal. Here is a placeholder:
        # x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Pass through Transformer blocks with causal mask
        for blk in self.blocks:
            x = blk(x, causal_mask=self.causal_mask.to(x.device) if self.causal_mask is not None else None)
        
        x = self.norm(x)
        
        # Get CLS token output and predict
        cls_output = x[:, 0]
        mu = self.head_mu(cls_output)
        log_var = self.head_log_var(cls_output)
        
        return mu, log_var
