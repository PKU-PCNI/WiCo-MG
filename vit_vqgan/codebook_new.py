"""
改进版 CodeBook：
- 支持 factorized codes（dim→lookup_dim→dim）
- L2 normalize 余弦距离
- 其余接口与旧版完全一致
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CodeBook(nn.Module):
    """
    Args:
        num_codebook_vectors : 码本向量数 K
        latent_dim           : 编码器输出/decoder 输入的 token 维度 (如 768)
        beta                 : commitment loss 系数
        lookup_dim           : factorized 低维查表空间 (论文用 32)
    """
    def __init__(
        self,
        num_codebook_vectors: int = 1024,
        latent_dim: int = 256,
        beta: float = 0.25,
        lookup_dim: int = 32,            # ★ 新增，默认为 32
    ):
        super().__init__()

        self.num_codebook_vectors = num_codebook_vectors
        self.latent_dim  = latent_dim
        self.lookup_dim  = lookup_dim
        self.beta        = beta

        # -------- ① dim → lookup_dim -----------
        self.enc2look = nn.Linear(latent_dim, lookup_dim)
        # -------- ② lookup_dim → dim -----------
        self.look2emb = nn.Linear(lookup_dim, latent_dim)

        # -------- 码本向量  K × lookup_dim -------
        self.codebook = nn.Embedding(num_codebook_vectors, lookup_dim)
        nn.init.normal_(self.codebook.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(self, z: torch.Tensor):
        """
        输入:
            z : [B, C, H, W]  —— encoder 输出经 1×1 conv 后
              （C == latent_dim）
        返回:
            z_q            : 量化后张量，与 z 同形状
            min_indices    : 最近邻索引 (B*H*W,)
            loss           : VQ + commitment
        """
        # -------- reshape → (BHW, C) ----------
        z_perm = z.permute(0, 2, 3, 1).contiguous()         # [B,H,W,C]
        z_flat = z_perm.view(-1, self.latent_dim)           # [BHW, C]

        # -------- dim → lookup_dim & L2 normalize ----------
        z_l = F.normalize(self.enc2look(z_flat), dim=1)     # [BHW, L]

        # -------- 码本同样 L2 normalize ----------
        codebook = F.normalize(self.codebook.weight, dim=1) # [K, L]

        # -------- 计算余弦距离 (欧氏同构) ----------
        dist = (z_l.pow(2).sum(1, keepdim=True)
               + codebook.pow(2).sum(1)
               - 2 * torch.matmul(z_l, codebook.t()))       # [BHW, K]

        # -------- 最近邻索引 ----------
        min_indices = torch.argmin(dist, dim=1)             # (BHW,)

        # -------- 查表 ----------
        z_q_l = self.codebook(min_indices)                  # [BHW, L]

        # -------- lookup_dim → latent_dim ----------
        z_q_flat = self.look2emb(z_q_l)                     # [BHW, C]

        # -------- 损失 (VQ + commitment) ----------
        loss = F.mse_loss(z_q_flat.detach(), z_flat) + \
               self.beta * F.mse_loss(z_q_flat, z_flat.detach())

        # -------- Straight-Through -----------
        z_q_flat = z_flat + (z_q_flat - z_flat).detach()

        # -------- reshape 回原格式 ----------
        z_q = z_q_flat.view_as(z_perm).permute(0, 3, 1, 2).contiguous()

        return z_q, min_indices, loss
