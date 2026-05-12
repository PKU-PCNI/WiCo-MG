import torch
import torch.nn as nn
from einops import rearrange, repeat
from .vit_components import PreNorm, Attention, FeedForward

# class Transformer(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
#                 PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
#             ]))

#     def forward(self, x):
#         for attn, ff in self.layers:
#             x = attn(x) + x
#             x = ff(x) + x
#         return x

# class VitEncoder(nn.Module):
#     def __init__(
#         self,
#         img_channels=3,
#         img_size=256,
#         patch_size=16,
#         dim=1024,
#         depth=6,
#         heads=8,
#         mlp_dim=2048,
#         dropout=0.1,
#         emb_dropout=0.1
#     ):
#         super().__init__()
#         self.img_size = img_size
#         num_patches = (img_size // patch_size) ** 2
#         patch_dim = img_channels * patch_size * patch_size

#         self.patch_size = patch_size
#         self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
#         self.patch_to_embedding = nn.Linear(patch_dim, dim)
#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.dropout = nn.Dropout(emb_dropout)

#         self.transformer = Transformer(dim, depth, heads, dim // heads, mlp_dim, dropout)

#         self.to_latent = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, dim)
#         )

#     def forward(self, img):
#         p = self.patch_size
#         x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
#         x = self.patch_to_embedding(x)
#         b, n, _ = x.shape

#         cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
#         x = torch.cat((cls_tokens, x), dim=1)
#         x += self.pos_embedding[:, :(n + 1)]
#         x = self.dropout(x)

#         x = self.transformer(x)
#         x = self.to_latent(x)
        
#         # 重新排列为特征图形式
#         x = x[:, 1:]  # 移除 CLS token
#         x = rearrange(x, 'b (h w) d -> b d h w', h=self.img_size//self.patch_size)
        
#         return x


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
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]) for _ in range(depth)
        ])

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class VitEncoder(nn.Module):
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
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size

        # 卷积替代 Linear Patch Embedding：输出 [B, dim, H', W']
        self.patch_embed = nn.Conv2d(
            in_channels=img_channels,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # 可学习的二维位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, dim, self.grid_size, self.grid_size))
        # self.register_buffer("pos_embedding", get_2d_sincos_pos_embed(dim, self.grid_size))
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer
        self.transformer = Transformer(dim, depth, heads, dim // heads, mlp_dim, dropout)

        # 输出前再 LayerNorm + Linear（可选）
        self.to_latent = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

    def forward(self, img):
        # 卷积提取 patch 特征
        x = self.patch_embed(img)                     # [B, dim, H', W']
        x = x + self.pos_embedding                    # 加 2D 位置编码
        x = self.dropout(x)

        # 展平通道维后进入 Transformer：[B, dim, H, W] → [B, HW, dim]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer(x)

        # to_latent: LayerNorm + Linear
        x = self.to_latent(x)

        # reshape 回 [B, dim, H, W] 格式
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.grid_size, w=self.grid_size)
        return x