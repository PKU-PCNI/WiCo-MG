import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Union



class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        mask = torch.tril(torch.ones(config.block_size,
                                     config.block_size))
        if hasattr(config, "n_unmasked"):
            mask[:config.n_unmasked, :config.n_unmasked] = 1
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        present = torch.stack((k, v))
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if layer_past is None:
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, present  # TODO: check that this does not break anything

class BidirectionalSelfAttention(nn.Module):
    """
    Standard (non-causal) multi-head self-attention layer.
    No causal mask. Suitable for encoder or cross-modal transformer.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # projections
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # dropout
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output proj
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head

    def forward(self, x):
        B, T, C = x.size()
        h = self.n_head
        head_dim = C // h

        # Linear projections and reshape
        k = self.key(x).view(B, T, h, head_dim).transpose(1, 2)  # [B, h, T, D]
        q = self.query(x).view(B, T, h, head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, h, head_dim).transpose(1, 2)

        # Attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))  # [B, h, T, T]
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # Attention output
        y = att @ v  # [B, h, T, D]
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # concat heads

        # Final projection
        y = self.resid_drop(self.proj(y))  # [B, T, C]
        return y, None




class MoEMLP_old(nn.Module):
    def __init__(self, n_embd, n_expert=6, n_shared=2, k=2):
        super().__init__()
        self.n_expert = n_expert
        self.n_shared = n_shared
        self.k = k  # top-k稀疏激活
        self.total_expert = n_expert + n_shared

        # 路由专家
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.GELU(),
                nn.Linear(4 * n_embd, n_embd),
            ) for _ in range(n_expert)
        ])
        # 共享专家
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.GELU(),
                nn.Linear(4 * n_embd, n_embd),
            ) for _ in range(n_shared)
        ])
        # 门控网络
        self.gate = nn.Linear(n_embd, self.total_expert)

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)  # [B*T, C] 所有 token 的嵌入展平后的表示
        gate_logits = self.gate(x_flat)  # [B*T, total_expert]
        gate_scores = F.softmax(gate_logits, dim=-1)
        topk_scores, topk_idx = torch.topk(gate_scores, self.k, dim=-1)  # [B*T, k]

        # 统计激活专家
        self.last_active_experts = set(topk_idx.cpu().numpy().flatten().tolist())

        # 路由到专家（deepseek风格，token-wise路由）
        expert_outputs = torch.zeros_like(x_flat)
        for i in range(self.k):
            expert_ids = topk_idx[:, i]  # [B*T]
            scores = topk_scores[:, i]   # [B*T]
            for expert_id in range(self.total_expert):
                mask = (expert_ids == expert_id) # [B*T]，哪些 token 被路由到这个专家
                if mask.sum() == 0:
                    continue
                input_i = x_flat[mask]
                if expert_id < self.n_expert:
                    out_i = self.experts[expert_id](input_i)
                else:
                    out_i = self.shared_experts[expert_id - self.n_expert](input_i)
                expert_outputs[mask] += out_i * scores[mask].unsqueeze(-1)
        return expert_outputs.view(B, T, C)

class MoEMLP(nn.Module):
    def __init__(self, n_embd, n_expert=6, n_shared=2, k=2):
        super().__init__()
        self.n_expert = n_expert
        self.n_shared = n_shared
        self.k = k  # top-k 稀疏激活
        self.total_expert = n_expert + n_shared

        # 路由专家（任务专家）
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.GELU(),
                nn.Linear(4 * n_embd, n_embd),
            ) for _ in range(n_expert)
        ])

        # 共享专家（所有 token 都通过）
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.GELU(),
                nn.Linear(4 * n_embd, n_embd),
            ) for _ in range(n_shared)
        ])

        # 门控网络：只为路由专家打分
        self.gate = nn.Linear(n_embd, n_expert)

        # 原 gate 现在只接受 token，本例中将 gate 拆分为 freq 和 token 分支
        self.freq_embed = nn.Sequential(
            nn.Linear(1, n_embd),
            nn.ReLU(),
            nn.Linear(n_embd, n_embd)
        )
        self.token_gate = nn.Linear(n_embd, n_embd)
        self.fuse_gate = nn.Linear(n_embd, n_expert)  # 最后输出 logits

    def forward(self, x, freq=None):
        B, T, C = x.shape
        x_flat = x.view(-1, C)  # [B*T, C]

        # Step 1: Gate 路由分数，仅用于任务专家
        if freq is not None:
            freq_input = freq.unsqueeze(-1).float()    # [B, 1]
            freq_emb = self.freq_embed(freq_input)     # [B, C]
            freq_emb = freq_emb.unsqueeze(1).expand(-1, T, -1).reshape(-1, C)
            gate_hidden = self.token_gate(x_flat) + freq_emb  # 条件调制（加而非拼接）
            gate_logits = self.fuse_gate(gate_hidden)         # [B*T, n_expert]
            gate_scores = F.softmax(gate_logits, dim=-1)
        else:
            gate_logits = self.gate(x_flat)  # [B*T, n_expert]
            gate_scores = F.softmax(gate_logits, dim=-1)  # [B*T, n_expert]

        topk_scores, topk_idx = torch.topk(gate_scores, self.k, dim=-1)  # [B*T, k]

        # Step 2: 路由专家处理
        expert_outputs = torch.zeros_like(x_flat)
        for i in range(self.k):
            expert_ids = topk_idx[:, i]      # 当前 token 第 i 高得分的专家编号
            scores = topk_scores[:, i]       # 对应得分
            for expert_id in range(self.n_expert):
                mask = (expert_ids == expert_id)
                if mask.sum() == 0:
                    continue
                input_i = x_flat[mask]
                out_i = self.experts[expert_id](input_i)
                expert_outputs[mask] += out_i * scores[mask].unsqueeze(-1)

        # Step 3: 所有 token 固定经过所有共享专家（平均后加到输出）
        shared_out = 0
        if self.n_shared > 0:
            for shared_expert in self.shared_experts:
                shared_out += shared_expert(x_flat)
            shared_out = shared_out / self.n_shared
        else:
            shared_out = 0  # 或者 torch.zeros_like(x_flat)，根据你的后续加法逻辑

        # Step 4: 加入共享专家结果
        output = expert_outputs + shared_out

        return output.view(B, T, C)


class TaskWiseMoEMLP_old(nn.Module):
    def __init__(self, n_embd, n_task=3, n_expert=6, n_shared=2, k=2):
        super().__init__()
        self.n_task = n_task
        self.n_expert = n_expert
        self.n_shared = n_shared
        self.k = k
        self.total_expert = n_expert + n_shared

        # 共用专家池
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.GELU(),
                nn.Linear(4 * n_embd, n_embd)
            ) for _ in range(n_expert)
        ])
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.GELU(),
                nn.Linear(4 * n_embd, n_embd)
            ) for _ in range(n_shared)
        ])

        # 每个任务一个门控网络
        self.gates = nn.ModuleList([
            nn.Linear(n_embd, self.total_expert) for _ in range(n_task)
        ])

    def forward(self, x):

        B, T, C = x.shape  # x: [B, T, C], 来自上一个 block
        x_flat = x.view(-1, C)  # [B*T, C]

        # 所有任务共用输入 token，但走不同门控
        outputs = []  # 每个任务一组输出 token，数量 = T

        for task_id in range(self.n_task):
            gate_logits = self.gates[task_id](x_flat)          # [B*T, total_expert]
            gate_scores = F.softmax(gate_logits, dim=-1)       # [B*T, total_expert]
            topk_scores, topk_idx = torch.topk(gate_scores, self.k, dim=-1)  # [B*T, k]

            # 用稀疏加权方式融合专家输出（非 token-wise，每组任务一个 gate）
            expert_output = torch.zeros_like(x_flat)

            for i in range(self.k):
                expert_ids = topk_idx[:, i]     # [B*T]
                scores = topk_scores[:, i]      # [B*T]

                for expert_id in range(self.total_expert):
                    mask = (expert_ids == expert_id)
                    if mask.sum() == 0:
                        continue
                    input_i = x_flat[mask]       # 被该 expert 接收的 token
                    if expert_id < self.n_expert:
                        out_i = self.experts[expert_id](input_i)
                    else:
                        out_i = self.shared_experts[expert_id - self.n_expert](input_i)
                    expert_output[mask] += out_i * scores[mask].unsqueeze(-1)

            outputs.append(expert_output.view(B, T, C))  # 每个任务一组输出 token

        return outputs  # list: [delay_token, angle_token, power_token] 一个列表

class TaskWiseMoEMLP(nn.Module):
    def __init__(self, n_embd, n_task=3, n_expert=6, n_shared=2, k=2):
        super().__init__()
        self.n_task = n_task
        self.n_expert = n_expert
        self.n_shared = n_shared
        self.k = k

        # 路由专家
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.GELU(),
                nn.Linear(4 * n_embd, n_embd)
            ) for _ in range(n_expert)
        ])

        # 共享专家（对所有 token 激活）
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.GELU(),
                nn.Linear(4 * n_embd, n_embd)
            ) for _ in range(n_shared)
        ])

        # 每个任务一个 gate，只针对路由专家
        self.gates = nn.ModuleList([
            nn.Linear(n_embd, n_expert) for _ in range(n_task)
        ])

    def forward(self, x, freq=None):
        B, T, C = x.shape
        x_flat = x.view(-1, C)  # [B*T, C]

        outputs = []

        for task_id in range(self.n_task):
            # Step 1: 路由专家的 gate
            gate_logits = self.gates[task_id](x_flat)          # [B*T, n_expert]
            gate_scores = F.softmax(gate_logits, dim=-1)
            topk_scores, topk_idx = torch.topk(gate_scores, self.k, dim=-1)  # [B*T, k]

            # Step 2: 路由专家稀疏激活
            expert_output = torch.zeros_like(x_flat)

            for i in range(self.k):
                expert_ids = topk_idx[:, i]
                scores = topk_scores[:, i]
                for expert_id in range(self.n_expert):
                    mask = (expert_ids == expert_id)
                    if mask.sum() == 0:
                        continue
                    input_i = x_flat[mask]
                    out_i = self.experts[expert_id](input_i)
                    expert_output[mask] += out_i * scores[mask].unsqueeze(-1)

            # Step 3: 所有共享专家都参与，输出求平均
            shared_out = 0
            if self.n_shared > 0:
                for shared_expert in self.shared_experts:
                    shared_out += shared_expert(x_flat)
                shared_out = shared_out / self.n_shared
            else:
                shared_out = 0

            # Step 4: 合并任务专家输出 + 共享专家输出
            task_output = expert_output + shared_out
            outputs.append(task_output.view(B, T, C))

        return outputs  # list: [delay_tokens, angle_tokens, power_tokens]

class TaskWiseMoEMLP_with_freq(nn.Module):
    def __init__(self, n_embd, n_task=3, n_expert=6, n_shared=2, k=2):
        super().__init__()
        self.n_task = n_task
        self.n_expert = n_expert
        self.n_shared = n_shared
        self.k = k

        # 路由专家
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.GELU(),
                nn.Linear(4 * n_embd, n_embd)
            ) for _ in range(n_expert)
        ])

        # 共享专家（所有 token 都通过）
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.GELU(),
                nn.Linear(4 * n_embd, n_embd)
            ) for _ in range(n_shared)
        ])

        # ====== Gate 相关 ======
        # 原始每任务 gate（当 freq=None 时使用）
        self.gates = nn.ModuleList([
            nn.Linear(n_embd, n_expert) for _ in range(n_task)
        ])

        # 频率调制版 gate：freq_embed + token_gate -> 每任务 fuse_gate
        self.freq_embed = nn.Sequential(
            nn.Linear(1, n_embd),
            nn.ReLU(),
            nn.Linear(n_embd, n_embd)
        )
        self.token_gate = nn.Linear(n_embd, n_embd)  # 共享的 token 分支
        self.fuse_gates = nn.ModuleList([            # 每个任务独立的融合投影
            nn.Linear(n_embd, n_expert) for _ in range(n_task)
        ])

    def forward(self, x, freq=None):
        """
        x: [B, T, C]
        freq: [B] 或 [B,] 的标量频率（float），当提供时用于调制 gate 打分。
        返回: list 长度为 n_task，每个元素是 [B, T, C]
        """
        B, T, C = x.shape
        x_flat = x.view(-1, C)  # [B*T, C]

        # 预计算共享专家输出（对所有任务相同）
        shared_out = torch.zeros_like(x_flat)
        for shared_expert in self.shared_experts:
            shared_out += shared_expert(x_flat)
        shared_out = shared_out / self.n_shared  # [B*T, C]

        # 若提供频率，先构造其嵌入并展开到 [B*T, C]
        if freq is not None:
            # freq: [B] -> [B, 1] -> [B, C] -> [B, 1, C] -> [B, T, C] -> [B*T, C]
            freq_input = freq.unsqueeze(-1).float()        # [B, 1]
            freq_emb = self.freq_embed(freq_input)         # [B, C]
            freq_emb = freq_emb.unsqueeze(1).expand(-1, T, -1).reshape(-1, C)  # [B*T, C]
            token_hidden = self.token_gate(x_flat)         # [B*T, C]
            fused_hidden = token_hidden + freq_emb         # [B*T, C]

        outputs = []

        for task_id in range(self.n_task):
            # ---- Step 1: 计算该任务的 gate 分数（只针对路由专家）----
            if freq is not None:
                gate_logits = self.fuse_gates[task_id](fused_hidden)  # [B*T, n_expert]
            else:
                gate_logits = self.gates[task_id](x_flat)             # [B*T, n_expert]
            gate_scores = F.softmax(gate_logits, dim=-1)              # [B*T, n_expert]

            topk_scores, topk_idx = torch.topk(gate_scores, self.k, dim=-1)  # [B*T, k]

            # ---- Step 2: 路由专家稀疏激活并加权聚合 ----
            expert_output = torch.zeros_like(x_flat)  # [B*T, C]
            for i in range(self.k):
                expert_ids = topk_idx[:, i]      # [B*T]
                scores = topk_scores[:, i]       # [B*T]
                for expert_id in range(self.n_expert):
                    mask = (expert_ids == expert_id)
                    if mask.any():
                        inp_i = x_flat[mask]                     # [N_i, C]
                        out_i = self.experts[expert_id](inp_i)   # [N_i, C]
                        expert_output[mask] += out_i * scores[mask].unsqueeze(-1)

            # ---- Step 3: 融合共享专家输出 ----
            task_output = expert_output + shared_out             # [B*T, C]
            outputs.append(task_output.view(B, T, C))

        return outputs


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = BidirectionalSelfAttention(config)
        # self.mlp = nn.Sequential(
        #     nn.Linear(config.n_embd, 4 * config.n_embd),
        #     nn.GELU(),  # nice
        #     nn.Linear(4 * config.n_embd, config.n_embd),
        #     nn.Dropout(config.resid_pdrop),
        # )
        # 替换为 MoE
        self.mlp = MoEMLP(config.n_embd, n_expert=config.n_expert, n_shared=config.n_shared, k=config.top_k)

    def forward(self, x, layer_past=None, return_present=False, freq=None):
        # TODO: check that training still works
        if return_present:
            assert not self.training
        # layer past: tuple of length two with B, nh, T, hs
        # attn, present = self.attn(self.ln1(x), layer_past=layer_past)
        attn, _ = self.attn(self.ln1(x))

        x = x + attn
        x = x + self.mlp(self.ln2(x),freq=freq)
        # if layer_past is not None or return_present:
        #     return x, present
        return x
    
class TaskWiseMoEBlock(nn.Module):
    """Transformer block with Bidirectional Self-Attention and Task-wise MoE"""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = BidirectionalSelfAttention(config)

        # 使用 Task-wise MoE 替代 MLP
        self.mlp = TaskWiseMoEMLP(
            n_embd=config.n_embd,
            n_task=config.n_task,
            n_expert=config.n_expert,
            n_shared=config.n_shared,
            k=config.top_k
        )

    def forward(self, x, freq=None):
        # x: [B, T, C]
        x = x + self.attn(self.ln1(x))[0]         # [B, T, C]
        shared_out = self.ln2(x)                  # [B, T, C]
        task_outputs = self.mlp(shared_out, freq)       # List of [B, T, C] per task
        return task_outputs                       # list length = n_task




class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, vocab_size, block_size, n_layer=12, n_head=8, n_embd=256,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0):
        super().__init__()
        config = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                           n_unmasked=n_unmasked)
        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))  # 512 x 1024
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, embeddings=None):
        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector

        if embeddings is not None:  # prepend explicit embeddings
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        return logits, None

class GPT_multitask_old_n_1(nn.Module):
    """Transformer model where the last block is a Task-wise MoE block"""

    def __init__(self, vocab_size, block_size, n_layer=12, n_head=8, n_embd=256,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0,
                 n_task=3, n_expert=8, n_shared=1, top_k=2):
        super().__init__()
        config = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                           n_unmasked=n_unmasked, n_task=n_task,
                           n_expert=n_expert, n_shared=n_shared, top_k=top_k)

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        self.blocks = nn.ModuleList([
            Block(config) for _ in range(config.n_layer - 1)
        ])
        self.final_block = TaskWiseMoEBlock(config)

        # decoder head（每个任务一个 head）
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.heads = nn.ModuleList([
            nn.Linear(config.n_embd, config.vocab_size, bias=False) for _ in range(config.n_task)
        ])

        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, embeddings=None, freq=None):

        token_embeddings = self.tok_emb(idx)
        if embeddings is not None:
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(token_embeddings + position_embeddings)

        for block in self.blocks:
            x = block(x,freq=freq)  # 每个 block 都处理输入

        task_outputs = self.final_block(x)  # list of [B, T, C] per task
        print(len(task_outputs), task_outputs[0].shape)

        # decoder heads
        logits_list = [self.heads[i](self.ln_f(task_out)) for i, task_out in enumerate(task_outputs)]
        return logits_list  # list of [B, T, vocab_size] per task


# ---- 聚合器：Conv1x1Aggregator ----
class Conv1x1Aggregator(nn.Module):
    """
    将中间 TaskWiseMoEBlock 的按任务输出（list of [B,T,C] 或 [B,H,W,C]）在任务维拼接后，
    通过 1x1 等效的逐位置 MLP 融合到 [B,*,C]。
    """
    def __init__(self, n_task: int, C: int, hidden_mult: float = 1.0):
        super().__init__()
        in_dim = n_task * C
        hidden = max(C, int(hidden_mult * C))
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, C, bias=True),
        )

    def forward(self, task_list: List[torch.Tensor]) -> torch.Tensor:
        if len(task_list) == 0:
            raise ValueError("Conv1x1Aggregator: empty task_list.")
        x0 = task_list[0]
        if x0.dim() == 3:
            # [B, T, C]
            x = torch.cat(task_list, dim=-1)     # [B, T, K*C]
            y = self.proj(x)                     # [B, T, C]
            return y
        elif x0.dim() == 4:
            # [B, H, W, C]
            x = torch.cat(task_list, dim=-1)     # [B, H, W, K*C]
            B, H, W, KC = x.shape
            x = x.view(B * H * W, KC)
            y = self.proj(x).view(B, H, W, -1)   # [B, H, W, C]
            return y
        else:
            raise ValueError("Conv1x1Aggregator expects per-task tensors of shape [B,T,C] or [B,H,W,C].")

class SoftTaskWeightedSumAggregator(nn.Module):
    def __init__(self, n_task, C):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_task, C))  # 每任务每通道一个权重

    def forward(self, xs):  # xs: list of [B,T,C], len = n_task
        B, T, C = xs[0].shape
        X = torch.stack(xs, dim=2)        # [B,T,Task,C]
        w = torch.softmax(self.w, dim=0)  # [Task,C]
        w = w.unsqueeze(0).unsqueeze(0)   # [1,1,Task,C]
        return (X * w).sum(dim=2)         # [B,T,C]


# ---- 主模型：任意排列 std/moe，最后必须是 moe ----
class GPT_multitask(nn.Module):
    """
    - 通过 block_plan 指定任意顺序的普通 Block("std") 与 TaskWiseMoEBlock("moe")
    - 中间出现的 "moe" 输出为 per-task list，需要先用 Conv1x1Aggregator 融合为张量后才能送入后续非 moe
    - 最后一个模块必须是 "moe"，并保留 per-task 输出以接入各任务 head
    """
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int = 12,
        n_head: int = 8,
        n_embd: int = 256,
        embd_pdrop: float = 0.0,
        resid_pdrop: float = 0.0,
        attn_pdrop: float = 0.0,
        n_unmasked: int = 0,
        n_task: int = 3,
        n_expert: int = 8,
        n_shared: int = 1,
        top_k: int = 2,
        block_plan: List[str] = None,   # 例如 ["std","std","moe","std","moe"]，最后必须是 "moe"
        agg_hidden_mult: float = 1.0,   # Conv1x1Aggregator 的中间宽度倍率
    ):
        super().__init__()

        # ---- 构建配置（假定你已有 GPTConfig 类）----
        self.config = GPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            embd_pdrop=embd_pdrop,
            resid_pdrop=resid_pdrop,
            attn_pdrop=attn_pdrop,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            n_unmasked=n_unmasked,
            n_task=n_task,
            n_expert=n_expert,
            n_shared=n_shared,
            top_k=top_k,
        )

        # 默认：前 n_layer-1 个 std + 最后 1 个 moe
        if block_plan is None:
            block_plan = ["std"] * (n_layer - 1) + ["moe"]

        if len(block_plan) == 0 or block_plan[-1] != "moe":
            raise ValueError("block_plan 的最后一个模块必须是 'moe' (TaskWiseMoEBlock)。")
        if "moe" not in block_plan:
            raise ValueError("block_plan 中需要至少包含一个 'moe'。")

        self.block_plan = block_plan
        self.n_task = n_task

        # ---- 嵌入层 ----
        self.tok_emb = nn.Embedding(self.config.vocab_size, self.config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.config.block_size, self.config.n_embd))
        self.drop = nn.Dropout(self.config.embd_pdrop)

        # ---- 构建模块序列 ----
        blocks = []
        for kind in block_plan:
            if kind == "std":
                blocks.append(Block(self.config))                # 你的标准 Transformer block
            elif kind == "moe":
                blocks.append(TaskWiseMoEBlock(self.config))     # 你的任务式 MoE block
            else:
                raise ValueError(f"Unknown block kind: {kind}")
        self.blocks = nn.ModuleList(blocks)

        # ---- 中间聚合器（只用 Conv1x1Aggregator）----
        self.aggregator = Conv1x1Aggregator(n_task=self.config.n_task, C=self.config.n_embd, hidden_mult=agg_hidden_mult)
        # self.aggregator = SoftTaskWeightedSumAggregator(n_task=self.config.n_task, C=self.config.n_embd)

        # ---- 任务解码头 ----
        self.ln_f = nn.LayerNorm(self.config.n_embd)
        self.heads = nn.ModuleList([
            nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
            for _ in range(self.config.n_task)
        ])

        self.block_size = self.config.block_size
        self.apply(self._init_weights)

        # siglip所用门控
        self.cond_proj = nn.Linear(n_embd, n_embd)  # 把 embeddings 压到 [B, C]
        self.gate = nn.Linear(n_embd, n_embd)       # 生成门控信号
        self.alpha = 0.001  # 控制门控强度

    # ---- 权重初始化 ----
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_block_size(self):
        return self.block_size

    @staticmethod
    def _is_task_list(x: Union[torch.Tensor, List[torch.Tensor]]) -> bool:
        return isinstance(x, (list, tuple))

    def forward(self, idx: torch.Tensor, embeddings: torch.Tensor = None, freq=None, path=None) -> List[torch.Tensor]:
        """
        idx: [B, Ttok]
        embeddings: 可选条件嵌入 [B, Tcond, C]，会拼接在 tokens 前
        freq: 透传给 Block / TaskWiseMoEBlock 的辅助信息（与你原实现一致）
        返回：list of logits，长度 = n_task，每个形状 [B, T, vocab_size]
        """
        token_embeddings = self.tok_emb(idx)  # [B, Ttok, C]
        # if embeddings is not None:
        #     token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)  # [B, Tcond+Ttok, C]
        if embeddings is not None:
            cond = self.cond_proj(embeddings.mean(dim=1))  # [B, C]
            cond = cond.unsqueeze(1).expand(-1, token_embeddings.size(1), -1)  # [B, Ttok, C]

            gate = torch.sigmoid(self.gate(token_embeddings))  # [B, Ttok, C]
            token_embeddings = token_embeddings + gate * cond * self.alpha  # 门控调制

        if path is not None:
            # print("path",path)
            # 支持 path 形状为 [B] / [B,1] / [B,Tp]
            if path.dim() == 1:
                path = path.unsqueeze(1)          # [B,1]
            path = path.to(idx.device)

            # 用已有 tok_emb 把 path 映射到 C 维，然后 cond_proj 压到语义向量
            # （几层：Embedding -> Linear -> sigmoid(gate) 融合）
            path_tok = self.tok_emb(path.clamp_min(0) % self.config.vocab_size)   # [B,Tp,C]
            path_vec = self.cond_proj(path_tok.mean(dim=1))                       # [B,C]
            path_vec = path_vec.unsqueeze(1).expand(-1, token_embeddings.size(1), -1)  # [B,Ttok,C]

            # gate = torch.sigmoid(self.gate(token_embeddings))  # [B,Ttok,C] 复用现有门控
            token_embeddings = token_embeddings + self.alpha * gate * path_vec

        T = token_embeddings.shape[1]
        if T > self.block_size:
            raise ValueError(f"Cannot forward, sequence length {T} exceeds block size {self.block_size}.")

        pos = self.pos_emb[:, :T, :]  # [1, T, C]
        x: Union[torch.Tensor, List[torch.Tensor]] = self.drop(token_embeddings + pos)

        final_task_outputs: List[torch.Tensor] = None
        num_blocks = len(self.blocks)

        for i, block in enumerate(self.blocks):
            is_last = (i == num_blocks - 1)
            next_is_std = (i < num_blocks - 1) and (not isinstance(self.blocks[i + 1], TaskWiseMoEBlock))

            if isinstance(block, TaskWiseMoEBlock):
                # 输入必须是张量；若上一步是 moe 并保留了 list，先聚合为张量
                if self._is_task_list(x):
                    x = self.aggregator(x)  # [B, T, C]（或 [B,H,W,C]）
                task_outputs = block(x, freq=freq)  # list of [B, T, C]
                if is_last:
                    # 最终保留 per-task 输出以接入各任务 head
                    final_task_outputs = task_outputs
                else:
                    # 若后继是 std，需要先融合成张量；若后继还是 moe（极少见），也建议融合，保证形状一致
                    x = self.aggregator(task_outputs)
            else:
                # 普通 Block 只接受张量
                if self._is_task_list(x):
                    x = self.aggregator(x)
                x = block(x, freq=freq)

        if final_task_outputs is None:
            raise RuntimeError("final_task_outputs 未生成。请确认 block_plan 的最后一个模块是 'moe'。")

        # 任务解码头
        logits_list = [self.heads[i](self.ln_f(task_out)) for i, task_out in enumerate(final_task_outputs)]
        return logits_list



    
class GPT_multitask_new(nn.Module):
    """
    - 通过 block_plan 指定任意顺序的普通 Block("std") 与 TaskWiseMoEBlock("moe")
    - 中间出现的 "moe" 输出为 per-task list，需要先用 Conv1x1Aggregator 融合为张量后才能送入后续非 moe
    - 最后一个模块必须是 "moe"，并保留 per-task 输出以接入各任务 head
    - 本版本：moe 之间的聚合模块不共享参数；为每个“moe->后续”过渡单独配置聚合器
    """
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int = 12,
        n_head: int = 8,
        n_embd: int = 256,
        embd_pdrop: float = 0.0,
        resid_pdrop: float = 0.0,
        attn_pdrop: float = 0.0,
        n_unmasked: int = 0,
        n_task: int = 3,
        n_expert: int = 8,
        n_shared: int = 1,
        top_k: int = 2,
        block_plan: List[str] = None,   # 例如 ["std","std","moe","std","moe"]，最后必须是 "moe"
        agg_hidden_mult: float = 1.0,
    ):
        super().__init__()

        # ---- 配置 ----
        self.config = GPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            embd_pdrop=embd_pdrop,
            resid_pdrop=resid_pdrop,
            attn_pdrop=attn_pdrop,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            n_unmasked=n_unmasked,
            n_task=n_task,
            n_expert=n_expert,
            n_shared=n_shared,
            top_k=top_k,
        )

        # 默认：前 n_layer-1 个 std + 最后 1 个 moe
        if block_plan is None:
            block_plan = ["std"] * (n_layer - 1) + ["moe"]

        if len(block_plan) == 0 or block_plan[-1] != "moe":
            raise ValueError("block_plan 的最后一个模块必须是 'moe' (TaskWiseMoEBlock)。")
        if "moe" not in block_plan:
            raise ValueError("block_plan 中需要至少包含一个 'moe'。")

        self.block_plan = block_plan
        self.n_task = n_task

        # ---- 嵌入层 ----
        self.tok_emb = nn.Embedding(self.config.vocab_size, self.config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.config.block_size, self.config.n_embd))
        self.drop = nn.Dropout(self.config.embd_pdrop)

        # ---- 构建模块序列 ----
        blocks = []
        for kind in block_plan:
            if kind == "std":
                blocks.append(Block(self.config))
            elif kind == "moe":
                blocks.append(TaskWiseMoEBlock(self.config))
            else:
                raise ValueError(f"Unknown block kind: {kind}")
        self.blocks = nn.ModuleList(blocks)

        # ---- 为每个“需要聚合”的 moe 位置单独创建聚合器 ----
        # moe_indices: 所有 moe 的索引；最后一个 moe 不需要聚合器（其输出保留为 list）
        self.moe_indices: List[int] = [i for i, k in enumerate(block_plan) if k == "moe"]
        self.last_moe_idx: int = self.moe_indices[-1]

        # 为每个“非最后的 moe 索引 i”创建一个专属聚合器，用于：
        # 1) 把前一个 moe 的 list 融合为张量后喂给当前块（当 i == 当前块索引-1 时使用）
        # 2) 把当前 moe 的 list 融合为张量喂给后续非 moe 块（当 i == 当前 moe 索引时使用）
        self.agg_after_moe = nn.ModuleDict({
            str(i): Conv1x1Aggregator(n_task=self.config.n_task, C=self.config.n_embd, hidden_mult=agg_hidden_mult)
            for i in self.moe_indices if i != self.last_moe_idx
        })

        # ---- 任务解码头 ----
        self.ln_f = nn.LayerNorm(self.config.n_embd)
        self.heads = nn.ModuleList([
            nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
            for _ in range(self.config.n_task)
        ])

        self.block_size = self.config.block_size
        self.apply(self._init_weights)

    # ---- 权重初始化 ----
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_block_size(self):
        return self.block_size

    @staticmethod
    def _is_task_list(x: Union[torch.Tensor, List[torch.Tensor]]) -> bool:
        return isinstance(x, (list, tuple))

    def forward(self, idx: torch.Tensor, embeddings: torch.Tensor = None, freq=None) -> List[torch.Tensor]:
        """
        idx: [B, Ttok]
        embeddings: 可选条件嵌入 [B, Tcond, C]，会拼接在 tokens 前
        freq: 透传给 Block / TaskWiseMoEBlock
        返回：list of logits，长度 = n_task，每个形状 [B, T, vocab_size]
        """
        token_embeddings = self.tok_emb(idx)  # [B, Ttok, C]
        if embeddings is not None:
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)  # [B, Tcond+Ttok, C]

        T = token_embeddings.shape[1]
        if T > self.block_size:
            raise ValueError(f"Cannot forward, sequence length {T} exceeds block size {self.block_size}.")

        pos = self.pos_emb[:, :T, :]  # [1, T, C]
        x: Union[torch.Tensor, List[torch.Tensor]] = self.drop(token_embeddings + pos)

        final_task_outputs: List[torch.Tensor] = None
        num_blocks = len(self.blocks)

        for i, block in enumerate(self.blocks):
            is_last = (i == num_blocks - 1)

            if isinstance(block, TaskWiseMoEBlock):
                # 若上一个是 moe，且它不是“最后的 moe”，其输出是 list，需要用“上一个 moe 的聚合器”融合成张量
                if self._is_task_list(x):
                    prev_idx = i - 1
                    key = str(prev_idx)
                    if key not in self.agg_after_moe:
                        raise RuntimeError(f"缺少聚合器：期望有 agg_after_moe['{key}'] 以融合前一个 moe 的输出。")
                    x = self.agg_after_moe[key](x)  # [B, T, C]

                # 当前 moe 产生 per-task list
                task_outputs = block(x, freq=freq)  # list of [B, T, C]

                if is_last:
                    # 最后一层 moe：保留 list，用于各任务 head
                    final_task_outputs = task_outputs
                else:
                    # 非最后的 moe：需要把 list 融合为张量以喂给后续块
                    key = str(i)
                    if key not in self.agg_after_moe:
                        raise RuntimeError(f"缺少聚合器：期望有 agg_after_moe['{key}'] 以融合当前 moe 的输出。")
                    x = self.agg_after_moe[key](task_outputs)  # [B, T, C]

            else:
                # 普通 Block 只接受张量；若输入是 list（来自前一 moe），先用“前一 moe 的聚合器”融合
                if self._is_task_list(x):
                    prev_idx = i - 1
                    key = str(prev_idx)
                    if key not in self.agg_after_moe:
                        raise RuntimeError(f"缺少聚合器：期望有 agg_after_moe['{key}'] 以融合前一个 moe 的输出。")
                    x = self.agg_after_moe[key](x)
                x = block(x, freq=freq)

        if final_task_outputs is None:
            raise RuntimeError("final_task_outputs 未生成。请确认 block_plan 的最后一个模块是 'moe'。")

        # 任务解码头
        logits_list = [self.heads[i](self.ln_f(task_out)) for i, task_out in enumerate(final_task_outputs)]
        return logits_list