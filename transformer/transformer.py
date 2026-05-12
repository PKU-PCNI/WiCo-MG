
# Importing Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.mingpt import GPT_multitask, GPT
from transformer.SigLIP2 import SigLIP2TokenExtractor
import os
import time


class VQGANTransformer(nn.Module):
    def __init__(
        self,
        vqgan: nn.Module, #CIR
        vqgan2: nn.Module, #RGB
        device: str = "cuda",
        sos_token: int = 0,
        pkeep: float = 0.5,
        block_size: int = 512,
        n_layer: int = 12,
        n_head: int = 16,
        n_embd: int = 1024,
    ):
        super().__init__()

        self.sos_token = sos_token
        self.device = device

        self.vqgan = vqgan #CIR
        self.vqgan2 = vqgan2 #RGB

        self.n_embd = n_embd

        self.transformer = GPT(
            vocab_size=self.vqgan.num_codebook_vectors,
            block_size=block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
        )

        self.pkeep = pkeep

    @torch.no_grad()
    def encode_to_z(self, x: torch.tensor) -> torch.tensor:
        """Processes the input batch ( containing images ) to encoder and returning flattened quantized encodings

        Args:
            x (torch.tensor): the input batch b*c*h*w

        Returns:
            torch.tensor: the flattened quantized encodings
        """
        quant_z, indices, _ = self.vqgan.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices

    @torch.no_grad()
    def encode_to_z2(self, x: torch.tensor) -> torch.tensor:
        """Processes the input batch ( containing images ) to encoder and returning flattened quantized encodings

        Args:
            x (torch.tensor): the input batch b*c*h*w

        Returns:
            torch.tensor: the flattened quantized encodings
        """
        quant_z, indices, _ = self.vqgan2.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices

    @torch.no_grad()
    def z_to_image(
        self, indices: torch.tensor, p1: int = 4, p2: int = 4, latent_channels: int = 512
    ) -> torch.Tensor:
        """Returns the decoded image from the indices for the codebook embeddings

        Args:
            indices (torch.tensor): the indices of the vectors in codebook to use for generating the decoder output
            p1 (int, optional): encoding size. Defaults to 16.
            p2 (int, optional): encoding size. Defaults to 16.

        Returns:
            torch.tensor: generated image from decoder
        """
        # print('indices',indices.shape) #[1,256]
        # print('self.vqgan.codebook.codebook(indices)', self.vqgan.codebook.codebook(indices).shape) #[1, 256, 256]
        # print('self.vqgan.codebook.codebook',self.vqgan.codebook.codebook) #Embedding(1024, 256)
        n = int(indices.shape[1] ** 0.5)  # indices shape 是 [B, n*n]
        ix_to_vectors = self.vqgan.codebook.codebook(indices).reshape(
            indices.shape[0], n, n, latent_channels
        )
        # print('ix_to_vectors',ix_to_vectors.shape) #[1,16,16,256]
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
        # print('ix_to_vectors',ix_to_vectors.shape) #[1,256,16,16]
        image = self.vqgan.decode(ix_to_vectors)
        # print('image',image.shape) #[1,1,256,256]
        return image

    def forward(self, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        """ x CIR  y RGB
        transformer model forward pass 

        Args:
            x (torch.tensor): Batch of images
        """

        # Getting the codebook indices of the image
        # print('x', x.shape) #torch.Size([16, 3, 128, 128]) 尺度变换之后的图像尺寸
        _, indices = self.encode_to_z(x) #获取 VQGAN 量化编码 射频
        # print('indices', indices.shape) #torch.Size([16, 64])

        _, indices_y = self.encode_to_z2(y)  # 获取 VQGAN 量化编码 RGB

        # sos tokens, this will be needed when we will generate new and unseen images
        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token #生成 SOS (Start of Sentence) Token
        sos_tokens = sos_tokens.long().to(self.device)

        # Generating a matrix of shape indices with 1s and 0s
        mask = torch.bernoulli( #生成随机 Mask 进行数据增强
            self.pkeep * torch.ones(indices.shape, device=indices.device)
        )  # torch.bernoulli([0.5 ... 0.5]) -> [1, 0, 1, 1, 0, 0] ; p(1) - 0.5
        mask = mask.round().to(dtype=torch.int64)

        # Generate a vector containing randomlly indices
        random_indices = torch.randint_like( #生成随机 Token 替换被 Mask 掉的部分
            indices, high=self.transformer.config.vocab_size
        )  # generating indices from the distribution

        """
        indices - [3, 56, 72, 67, 45, 53, 78, 90]
        mask - [1, 1, 0, 0, 1, 1, 1, 0]
        random_indices - 15, 67, 27, 89, 92, 40, 91, 10]

        new_indices - [ 3, 56,  0,  0, 45, 53, 78,  0] + [ 0,  0, 27, 89,  0,  0,  0, 10] => [ 3, 56, 27, 89, 45, 53, 78, 10]
        """
        new_indices = mask * indices + (1 - mask) * random_indices
        # print('new_indices', new_indices.shape) #torch.Size([16, 64])   16是batch size

        # Adding sos ( start of sentence ) token
        new_indices = torch.cat((sos_tokens, new_indices), dim=1) #拼接 SOS Token
        # print('new_indices', new_indices.shape) #torch.Size([16, 65])

        target = indices
        # print('target',target.shape) #torch.Size([16, 64])

        # new_indices = new_indices.long()
        # logits, _ = self.transformer(new_indices[:, :-1], context=indices_y) #送入 Transformer 计算 logits
        # logits, _ = self.transformer(torch.cat((indices_y, new_indices[:, :-1]), dim=1))  # 送入 Transformer 计算 logits
        logits, _ = self.transformer(indices_y)  # 送入 Transformer 计算 logits

        # print('logits', logits.shape) #torch.Size([16, 64, 512])  512是latent channel的维度

        return logits, target #在TransformerTrainer里算交叉熵损失

    # 需要确认transformer的输入输出的维度是多少，mask的机制是如何运作的，然后应该是要改generate那个函数

    # def forward(self, cir_imgs: torch.Tensor, rgb_imgs: torch.Tensor) -> torch.Tensor:
    #     """
    #     cir_imgs: CIR 图像 batch
    #     rgb_imgs: RGB 图像 batch
    #     返回 logits 和 target
    #     """
    #     # 1. 编码得到 CIR 和 RGB 的 codebook indices
    #     _, cir_indices = self.encode_to_z(cir_imgs)   # [B, L_cir]
    #     _, rgb_indices = self.encode_to_z2(rgb_imgs)  # [B, L_rgb]

    #     # 2. 构造 [sos] token
    #     sos_tokens = torch.full(
    #         (cir_indices.size(0), 1),
    #         fill_value=self.sos_token,
    #         dtype=cir_indices.dtype,
    #         device=cir_indices.device
    #     )  # [B, 1]

    #     # 3. 拼接输入序列：[RGB_token, [sos], CIR_token[:-1]]
    #     input_indices = torch.cat([rgb_indices, sos_tokens, cir_indices[:, :-1]], dim=1)  # [B, L_rgb + 1 + L_cir - 1]

    #     # 4. 目标序列：CIR_token
    #     target = cir_indices  # [B, L_cir]

    #     # 5. 构造 attention mask，保证每个 token 只能看到前面的 token
    #     # seq_len = input_indices.shape[1]
    #     # attn_mask = torch.tril(torch.ones((seq_len, seq_len), device=input_indices.device)).unsqueeze(0)  # [1, seq_len, seq_len]

    #     # 6. 送入 transformer，内部自动做了自回归掩码（即 causal mask），不需要你手动传递
    #     logits, _ = self.transformer(input_indices)

    #     # 7. 只取 CIR 部分的 logits 作为输出（与 target 对齐）
    #     logits = logits[:, -target.shape[1]:, :]  # [B, L_cir, vocab_size]

    #     return logits, target
    

    
    def top_k_logits(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """
        在采样阶段，限制输出概率最高的 k 个 token，避免低概率 token 影响生成质量
        Args:
            logits (torch.Tensor): predictions from the transformer
            k (int): returning k highest values

        Returns:
            torch.Tensor: retuning tensor of same dimension as input keeping the top k entries
        """
        v, ix = torch.topk(logits, k) #取 logits 中前 k 大的值 v 及其索引 ix。
        out = logits.clone()
        out[out < v[..., [-1]]] = -float( # 只保留前 k 大的 logits，其他设为负无穷
            "inf"
        )  # Setting all values except in topk to inf
        return out

    @torch.no_grad()
    def sample(
        self,
        rgb_indices: torch.Tensor,     # 输入 RGB token 索引 [B, L_rgb]
        steps: int = 64                # 要生成的 CIR token 个数（即 L_cir）
    ) -> torch.Tensor:
        """
        使用非自回归 Transformer，直接从 RGB token 预测 CIR token
        Args:
            rgb_indices: [B, L_rgb]，RGB 图像编码后的离散 token 序列
            steps: 目标 CIR token 数量（如 64）

        Returns:
            cir_indices: [B, steps]，预测出的 CIR token 索引
        """
        self.transformer.eval()

        # 一次性前向推理，输出 CIR 的 logits
        logits, _ = self.transformer(rgb_indices)  # [B, steps, vocab_size]
        print(f"logits shape: {logits.shape}")
        probs = F.softmax(logits, dim=-1)       # [B, steps, vocab_size]

        # 多项式采样，每个位置选一个 token
        cir_indices = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1)
        cir_indices = cir_indices.view(probs.size(0), steps)  # [B, steps]

        self.transformer.train()
        return cir_indices
    
    # def sample( #给定部分 token，利用 Transformer 自回归生成完整 token 序列
    #     self,
    #     x: torch.Tensor,
    #     c: torch.Tensor,
    #     steps: int = 256,
    #     temperature: float = 1.0,
    #     top_k: int = 100,
    # ) -> torch.Tensor:
    #     """Generating sample indices from the transformer

    #     Args:
    #         x (torch.Tensor): the batch of images
    #         c (torch.Tensor): sos token 
    #         steps (int, optional): the lenght of indices to generate. Defaults to 256.
    #         temperature (float, optional): hyperparameter for minGPT model. Defaults to 1.0.
    #         top_k (int, optional): keeping top k entries. Defaults to 100.

    #     Returns:
    #         torch.Tensor: _description_
    #     """

    #     self.transformer.eval()
    #     # print('x1', x.shape) #[4, 0]

    #     x = torch.cat((x, c), dim=1)  # Appending sos token
    #     # print('x1',x.shape) #[4, 1]

    #     for k in range(steps):
    #         logits, _ = self.transformer(x)  # Getting the predicted indices
    #         # print('logits', logits.shape) #[4, 52, 512]第二维逐渐加1

    #         logits = (
    #             logits[:, -1, :] / temperature
    #         )  # Getting the last prediction and scaling it by temperature
    #         # print('logits', logits.shape) #[4, 512]

    #         if top_k is not None:
    #             logits = self.top_k_logits(logits, top_k) #使用 top_k_logits 只保留前 k 个可能的 token
    #         # print('logits', logits.shape) #[4, 512]

    #         probs = F.softmax(logits, dim=-1)
    #         # print('probs', probs.shape)

    #         ix = torch.multinomial( # 按概率采样一个 token
    #             probs, num_samples=1
    #         )  # Note : not sure what's happening here
    #         # print('ix',ix)

    #         x = torch.cat((x, ix), dim=1) # 添加到序列中，继续循环
    #         # print('x2', x.shape) #第二维逐渐加1

    #     # x = x[:, c.shape[1] :]  # Removing the sos token
    #     x = x[:, steps+1:]
    #     self.transformer.train()
    #     return x

    @torch.no_grad()
    def log_images(self, x:torch.Tensor):
        """ Generating images using the transformer and decoder. Also uses encoder to complete partial images.   
        通过 Transformer 进行 条件和无条件的图像生成，并返回原始输入、重建图像、条件生成图像和自由生成图像
        Args:
            x (torch.Tensor): batch of images

        Returns:
            Retures the input and generated image in dictionary and in a simple concatenated image

            | 原始输入 | 重建图像 | 半条件生成 | 自由生成 |
            |----------|----------|------------|----------|
            |  [dog]   |  [dog]   | [狗头+生成身体] | [随机生成动物] |
        """
        log = dict() # 创建日志字典存储不同生成结果

        _, indices = self.encode_to_z(x) # Getting the indices of the quantized encoding
        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token # 创建序列起始符
        sos_tokens = sos_tokens.long().to(self.device) # 转换为整数类型并送设备

        start_indices = indices[:, : indices.shape[1] // 2] # 取前50%潜在编码作为条件
        sample_indices = self.sample( # 自回归生成后半段编码
            start_indices, sos_tokens, steps=indices.shape[1] - start_indices.shape[1]
        )
        # print("sample_indices",sample_indices.shape)
        half_sample = self.z_to_image(sample_indices, latent_channels=self.n_embd) # 解码为图像
        # print("half_sample", half_sample.shape)

        start_indices = indices[:, :0] # 空条件（生成完全自主）
        sample_indices = self.sample(start_indices, sos_tokens, steps=indices.shape[1])
        full_sample = self.z_to_image(sample_indices, latent_channels=self.n_embd) # 解码为图像

        x_rec = self.z_to_image(indices, latent_channels=self.n_embd) # 完整编码直接解码（验证重建能力）

        log["input"] = x # 原始输入
        log["rec"] = x_rec # 重建结果
        log["half_sample"] = half_sample # 条件生成结果
        log["full_sample"] = full_sample # 自由生成结果

        return log, torch.concat((x, x_rec, half_sample, full_sample)) # 拼接可视化

    # def load_checkpoint(self, path):
    #     """Loads the checkpoint from the given path."""

    #     self.load_state_dict(torch.load(path))

    def load_checkpoint(self, path: str, device: str = "cuda"):
        checkpoint = torch.load(path, map_location=device)  # 加载到指定设备
        self.load_state_dict(checkpoint)  # 加载模型权重
        print(f"Checkpoint loaded from {path} to {device}")

    def save_checkpoint(self, path):
        """Saves the checkpoint to the given path."""

        torch.save(self.state_dict(), path)


class multiVQGANTransformer(nn.Module):
    def __init__(
        self,
        vqgan: torch.nn.Module,
        vqgan1: torch.nn.Module,
        vqgan2: torch.nn.Module,
        vqgan3: torch.nn.Module,
        vqgan4: torch.nn.Module,
        device: str = "cuda",
        sos_token: int = 0,
        pkeep: float = 0.5,
        block_size: int = 512,
        n_layer: int = 12,
        n_head: int = 16,
        n_embd: int = 1024,
    ):
        super().__init__()

        self.sos_token = sos_token
        self.device = device

        self.vqgan = vqgan #RGB
        self.vqgan1 = vqgan1
        self.vqgan2 = vqgan2
        self.vqgan3 = vqgan3
        self.vqgan4 = vqgan4

        self.n_embd = n_embd

        # self.transformer = GPT_multitask(
        #     vocab_size=self.vqgan.num_codebook_vectors,
        #     block_size=block_size,
        #     n_layer=n_layer,
        #     n_head=n_head,
        #     n_embd=n_embd,
        #     n_task=4,  # 任务数量
        # )

        self.transformer = GPT_multitask(
            vocab_size=self.vqgan.num_codebook_vectors,
            block_size=block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            n_task=4,  # 任务数量
            n_expert=8,
            n_shared=1,
            top_k=2,
            block_plan=["std","std","std","std","std","std","moe","moe"],  # 最后必须是 moe
            agg_hidden_mult=1.5,                         # 可调
        )

        # self.siglip2_tokens = SigLIP2TokenExtractor(
        #     variant="so400m-384",   # 一行切换：base / large / so400m / g ……
        #     proj_dim=n_embd,           # 投到与你 Transformer 的 d_model 对齐，例如 512
        #     freeze=True,            # 只做前向提特征
        #     include_cls=True,       # 也可 False，只用 patch token
        #     norm_tokens=False,
        #     use_fp16=True,          # 推理更省
        # )

        self.siglip2_tokens = SigLIP2TokenExtractor(
            variant="base",                  # 或 model_name="google/siglip2-base-patch16-224"
            local_dir=os.environ.get("SIGLIP2_LOCAL_DIR", None),  # 指向你完整下载的目录可离线
            cache_dir=os.path.expanduser("~/hf_cache/transformers"),
            local_files_only=False,          # 设 True 可彻底离线（local_dir 必须完整）
            proj_dim=n_embd,                   # 保持原 hidden_size (Base=768)
            freeze=False, include_cls=False, norm_tokens=False, use_amp=True
        ).cuda()


        self.pkeep = pkeep

    @torch.no_grad()
    def encode_to_z(self, x: torch.tensor) -> torch.tensor:

        quant_z, indices, _ = self.vqgan.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices

    @torch.no_grad()
    def encode_to_z1(self, x: torch.tensor) -> torch.tensor:

        quant_z, indices, _ = self.vqgan1.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices

    @torch.no_grad()
    def encode_to_z2(self, x: torch.tensor) -> torch.tensor:

        quant_z, indices, _ = self.vqgan2.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices

    @torch.no_grad()
    def encode_to_z3(self, x: torch.tensor) -> torch.tensor:

        quant_z, indices, _ = self.vqgan3.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices

    @torch.no_grad()
    def encode_to_z4(self, x: torch.tensor) -> torch.tensor:

        quant_z, indices, _ = self.vqgan4.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices



    @torch.no_grad()
    def z_to_image1(
        self, indices: torch.tensor, p1: int = 4, p2: int = 4, latent_channels: int = 512
    ) -> torch.Tensor:
        # latent_channels = self.n_embd
        # print(latent_channels)
        n = int(indices.shape[1] ** 0.5)  # indices shape 是 [B, n*n]
        ix_to_vectors = self.vqgan1.codebook.codebook(indices).reshape(
            indices.shape[0], n, n, latent_channels
        )
        # print('ix_to_vectors',ix_to_vectors.shape) #[1,16,16,256]
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
        # print('ix_to_vectors',ix_to_vectors.shape) #[1,256,16,16]
        image = self.vqgan1.decode(ix_to_vectors)
        # print('image',image.shape) #[1,1,256,256]
        return image
    
    @torch.no_grad()
    def z_to_image2(
        self, indices: torch.tensor, p1: int = 4, p2: int = 4, latent_channels: int = 512
    ) -> torch.Tensor:
        # latent_channels = self.n_embd
        n = int(indices.shape[1] ** 0.5)  # indices shape 是 [B, n*n]
        ix_to_vectors = self.vqgan2.codebook.codebook(indices).reshape(
            indices.shape[0], n, n, latent_channels
        )
        # print('ix_to_vectors',ix_to_vectors.shape) #[1,16,16,256]
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
        # print('ix_to_vectors',ix_to_vectors.shape) #[1,256,16,16]
        image = self.vqgan2.decode(ix_to_vectors)
        # print('image',image.shape) #[1,1,256,256]
        return image
    
    @torch.no_grad()
    def z_to_image3(
        self, indices: torch.tensor, p1: int = 4, p2: int = 4, latent_channels: int = 512
    ) -> torch.Tensor:
        # latent_channels = self.n_embd
        n = int(indices.shape[1] ** 0.5)  # indices shape 是 [B, n*n]
        ix_to_vectors = self.vqgan3.codebook.codebook(indices).reshape(
            indices.shape[0], n, n, latent_channels
        )
        # print('ix_to_vectors',ix_to_vectors.shape) #[1,16,16,256]
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
        # print('ix_to_vectors',ix_to_vectors.shape) #[1,256,16,16]
        image = self.vqgan3.decode(ix_to_vectors)
        # print('image',image.shape) #[1,1,256,256]
        return image
    
    @torch.no_grad()
    def z_to_image4(
        self, indices: torch.tensor, p1: int = 4, p2: int = 4, latent_channels: int = 512
    ) -> torch.Tensor:
        # latent_channels = self.n_embd
        
        n = int(indices.shape[1] ** 0.5)  # indices shape 是 [B, n*n]
        ix_to_vectors = self.vqgan4.codebook.codebook(indices).reshape(
            indices.shape[0], n, n, latent_channels
        )
        # print('ix_to_vectors',ix_to_vectors.shape) #[1,16,16,256]
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
        # print('ix_to_vectors',ix_to_vectors.shape) #[1,256,16,16]
        image = self.vqgan4.decode(ix_to_vectors)
        # print('image',image.shape) #[1,1,256,256]
        return image

        

    def forward(self, x1:torch.Tensor, x2:torch.Tensor, x3:torch.Tensor, x4:torch.Tensor, y:torch.Tensor, freq: torch.Tensor = None, path: torch.Tensor = None) -> torch.Tensor:
        """ x CIR  y RGB
        transformer model forward pass 

        Args:
            x (torch.tensor): Batch of images
        """
        # print("params(M):", self.siglip2_tokens.count_params()/1e6)
        # print('x', x.shape) #torch.Size([16, 3, 128, 128]) 尺度变换之后的图像尺寸

        

        _, indices1 = self.encode_to_z1(x1) #获取 VQGAN 量化编码 射频
        _, indices2 = self.encode_to_z2(x2)
        _, indices3 = self.encode_to_z3(x3)
        _, indices4 = self.encode_to_z4(x4)
        targets = [indices1, indices2, indices3, indices4]
        # print('indices', indices.shape) #torch.Size([16, 64])

        _, indices_y = self.encode_to_z(y)  # 获取 VQGAN 量化编码 RGB

        siglip_tokens = self.siglip2_tokens(y)  # [B, T_s, 512]
        # print('siglip_tokens', siglip_tokens.shape)  # [4, 196, 256]

        if freq is not None:
            if path is not None:
                logits = self.transformer(indices_y, embeddings=siglip_tokens, freq=freq, path=path)  # GPT的实例
            else:
                logits = self.transformer(indices_y, embeddings=siglip_tokens, freq=freq)  # GPT的实例
        else:
            logits = self.transformer(indices_y)

        # print('logits', logits.shape) #torch.Size([16, 64, 512])  512是latent channel的维度

        # end_time = time.time()  # 记录结束时间
        # print(f"Execution time: {end_time - start_time:.4f} seconds")

        return logits, targets #在TransformerTrainer里算交叉熵损失

   

    
    def top_k_logits(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """
        在采样阶段，限制输出概率最高的 k 个 token，避免低概率 token 影响生成质量
        Args:
            logits (torch.Tensor): predictions from the transformer
            k (int): returning k highest values

        Returns:
            torch.Tensor: retuning tensor of same dimension as input keeping the top k entries
        """
        v, ix = torch.topk(logits, k) #取 logits 中前 k 大的值 v 及其索引 ix。
        out = logits.clone()
        out[out < v[..., [-1]]] = -float( # 只保留前 k 大的 logits，其他设为负无穷
            "inf"
        )  # Setting all values except in topk to inf
        return out

    @torch.no_grad()
    def sample(
        self,
        RGB: torch.Tensor,
        rgb_indices: torch.Tensor,     # 输入 RGB token 索引 [B, L_rgb]
        steps: int = 64,                # 要生成的 CIR token 个数（即 L_cir）
        freq: torch.Tensor = None,      # 可选的频率信息
        path: torch.Tensor = None      # 可选的路径信息
    ) -> torch.Tensor:

        self.transformer.eval()

        # 一次性前向推理，输出 CIR 的 logits
        # if freq is not None:
        #     logits_list = self.transformer(rgb_indices, freq=freq)
        # else:
        #     logits_list = self.transformer(rgb_indices)  # [B, steps, vocab_size]

        siglip_tokens = self.siglip2_tokens(RGB)  # [B, T_s, 512]

        if freq is not None:
            if path is not None:
                logits_list = self.transformer(rgb_indices, embeddings=siglip_tokens, freq=freq, path=path)  # GPT的实例
            else:
                # start_time = time.time()
                logits_list = self.transformer(rgb_indices, embeddings=siglip_tokens, freq=freq)  # GPT的实例
                # end_time = time.time()
                # print(f"代码执行时间: {end_time - start_time:.6f} 秒")
        else:
            logits_list = self.transformer(rgb_indices)

        
        cir_indices_list = []
        for logits in logits_list:

            probs = F.softmax(logits, dim=-1)  # [B, steps, vocab]
            sampled = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1)
            sampled = sampled.view(probs.size(0), steps)  # [B, steps]
            cir_indices_list.append(sampled)

        self.transformer.train()
        return cir_indices_list  # list of [B, steps]


    @torch.no_grad()
    def log_images(self, x:torch.Tensor):
        """ Generating images using the transformer and decoder. Also uses encoder to complete partial images.   
        通过 Transformer 进行 条件和无条件的图像生成，并返回原始输入、重建图像、条件生成图像和自由生成图像
        Args:
            x (torch.Tensor): batch of images

        Returns:
            Retures the input and generated image in dictionary and in a simple concatenated image

            | 原始输入 | 重建图像 | 半条件生成 | 自由生成 |
            |----------|----------|------------|----------|
            |  [dog]   |  [dog]   | [狗头+生成身体] | [随机生成动物] |
        """
        log = dict() # 创建日志字典存储不同生成结果

        _, indices = self.encode_to_z(x) # Getting the indices of the quantized encoding
        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token # 创建序列起始符
        sos_tokens = sos_tokens.long().to(self.device) # 转换为整数类型并送设备

        start_indices = indices[:, : indices.shape[1] // 2] # 取前50%潜在编码作为条件
        sample_indices = self.sample( # 自回归生成后半段编码
            start_indices, sos_tokens, steps=indices.shape[1] - start_indices.shape[1]
        )
        # print("sample_indices",sample_indices.shape)
        half_sample = self.z_to_image(sample_indices, latent_channels=self.n_embd) # 解码为图像
        # print("half_sample", half_sample.shape)

        start_indices = indices[:, :0] # 空条件（生成完全自主）
        sample_indices = self.sample(start_indices, sos_tokens, steps=indices.shape[1])
        full_sample = self.z_to_image(sample_indices, latent_channels=self.n_embd) # 解码为图像

        x_rec = self.z_to_image(indices, latent_channels=self.n_embd) # 完整编码直接解码（验证重建能力）

        log["input"] = x # 原始输入
        log["rec"] = x_rec # 重建结果
        log["half_sample"] = half_sample # 条件生成结果
        log["full_sample"] = full_sample # 自由生成结果

        return log, torch.concat((x, x_rec, half_sample, full_sample)) # 拼接可视化

    # def load_checkpoint(self, path):
    #     """Loads the checkpoint from the given path."""

    #     self.load_state_dict(torch.load(path))

    def load_checkpoint(self, path: str, device: str = "cuda"):
        checkpoint = torch.load(path, map_location=device)  # 加载到指定设备
        self.load_state_dict(checkpoint)  # 加载模型权重
        print(f"！！！！！！！！Checkpoint loaded from {path} to {device}")

    def save_checkpoint(self, path):
        """Saves the checkpoint to the given path."""

        torch.save(self.state_dict(), path)
