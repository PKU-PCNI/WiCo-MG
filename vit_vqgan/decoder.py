import torch
import torch.nn as nn
from einops import rearrange
from .vit_components import PreNorm, Attention, FeedForward


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    返回 shape 为 [1, embed_dim, grid_size, grid_size] 的不可学习位置编码（2D sin-cos）
    """
    h, w = grid_size, grid_size
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing='ij')  # 2 x H x W
    grid = torch.stack(grid, dim=0)  # [2, H, W]
    grid = grid.reshape(2, 1, h, w)
    pos_embed = []

    for i in range(embed_dim // 4):
        div_term = 10000 ** (4 * i / embed_dim)
        pos_embed.append(torch.sin(grid[0] / div_term))
        pos_embed.append(torch.cos(grid[0] / div_term))
        pos_embed.append(torch.sin(grid[1] / div_term))
        pos_embed.append(torch.cos(grid[1] / div_term))

    pos_embed = torch.cat(pos_embed, dim=0).unsqueeze(0)  # [1, dim, H, W]
    return pos_embed

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    
    
# class VitDecoder(nn.Module):
#     def __init__(
#         self,
#         img_channels=3,
#         img_size=256,
#         patch_size=16,
#         dim=1024,
#         depth=6,
#         heads=8,
#         mlp_dim=2048,
#         dropout=0.1
#     ):
#         super().__init__()
#         self.patch_size = patch_size
#         self.img_size = img_size
#         num_patches = (img_size // patch_size) ** 2
        
#         self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
#         self.transformer = Transformer(dim, depth, heads, dim // heads, mlp_dim, dropout)
        
#         # 修改 to_pixels 的结构，确保最后一层有权重
#         self.to_pixels = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, patch_size * patch_size * img_channels),
#             nn.Linear(patch_size * patch_size * img_channels, 
#                      patch_size * patch_size * img_channels),  # 添加一个带权重的线性层
#             nn.Tanh()
#         )

#     def forward(self, x):
#         # 输入 x 形状: [b, c, h, w]
#         x = rearrange(x, 'b c h w -> b (h w) c')
#         x = x + self.pos_embedding
#         x = self.transformer(x)
#         x = self.to_pixels(x)
        
#         # 重建图像
#         x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
#                      h=self.img_size//self.patch_size, 
#                      p1=self.patch_size, 
#                      p2=self.patch_size)
#         return x
    

class VitDecoder(nn.Module):
    def __init__(
        self,
        img_channels=3,
        img_size=256,
        patch_size=16,
        dim=1024,
        depth=6,
        heads=8,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    ):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size  # 例如 16

        # 2D位置编码，与输入尺寸对应：[1, dim, H', W']
        self.pos_embedding = nn.Parameter(torch.randn(1, dim, self.grid_size, self.grid_size))
        # self.register_buffer("pos_embedding", get_2d_sincos_pos_embed(dim, self.grid_size))
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer（输入格式为 flatten 后的 token）
        self.transformer = Transformer(dim, depth, heads, dim // heads, mlp_dim, dropout)

        # 输出图像的上采样模块
        self.to_pixels = nn.Sequential(
            nn.Upsample(scale_factor=patch_size, mode='bilinear', align_corners=False),  # → [b, dim, 256, 256]
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 4, img_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # 输入 x: [b, 1024, 16, 16]
        x = x + self.pos_embedding                    # 加二维位置编码 → [b, 1024, 16, 16]
        x = self.dropout(x)

        x = rearrange(x, 'b c h w -> b (h w) c')      # → [b, 256, 1024]
        x = self.transformer(x)                       # → [b, 256, 1024]
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.grid_size, w=self.grid_size)  # → [b, 1024, 16, 16]

        x = self.to_pixels(x)                          # → [b, 3, 256, 256]
        # x = torch.tanh(x)
        return x
