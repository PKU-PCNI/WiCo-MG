# import torch
# import torch.nn as nn
# from typing import Optional, Tuple, Union, List, Dict
# from transformers import AutoModel, AutoProcessor
# from torchvision.transforms.functional import to_pil_image
# import os


# SIGLIP2_PRESETS: Dict[str, str] = {
#     # 你可按需补充更多官方/社区权重别名
#     # 官方常见：
#     "base-224":    "google/siglip2-base-patch16-224",
#     "base-384":    "google/siglip2-base-patch16-384",
#     "large-224":   "google/siglip2-large-patch16-224",
#     "large-384":   "google/siglip2-large-patch16-384",
#     "so400m-384":  "google/siglip2-so400m-patch14-384",
#     "so400m-448":  "google/siglip2-so400m-patch14-448",
#     "giant-384":   "google/siglip2-gpatch14-384",
#     "giant-448":   "google/siglip2-gpatch14-448",
#     # 兼容写法（简写）
#     "base":        "google/siglip2-base-patch16-224",
#     "large":       "google/siglip2-large-patch16-384",
#     "so400m":      "google/siglip2-so400m-patch14-384",
#     "g":           "google/siglip2-gpatch14-448",
# }


# class SigLIP2TokenExtractor(nn.Module):
#     """
#     将 RGB 图像批 y: [B,3,H,W] 输入 SigLIP-2 视觉塔，输出一串 patch 特征 token（可选含 CLS），
#     并可线性投影到统一维度以便作为下游 Transformer 的额外输入。

#     版本选择方式（任选其一）：
#       1) 传入 variant=<别名>，如 "base", "large", "so400m", "g" 或 "base-224"/"so400m-384" 等；
#       2) 直接传入 model_name=<完整HF权重名>，如 "google/siglip2-base-patch16-224"。

#     Args:
#         variant:   预设别名（见 SIGLIP2_PRESETS）。若提供，则忽略 model_name。
#         model_name: 完整的 Hugging Face 权重名（与 variant 二选一）。
#         proj_dim:  输出 token 的维度；None 则保持视觉塔 hidden_size。
#         freeze:    是否冻结 SigLIP-2 权重（默认 True，仅推理提特征更省显存）。
#         include_cls: 返回序列是否包含 CLS token（默认 True）。
#         norm_tokens: 是否逐 token 做 L2 归一化（默认 False）。
#         use_fp16:  是否在特征提取中使用 autocast(fp16/bf16 由外部AMP环境决定)。
#     """

#     def __init__(
#         self,
#         *,
#         variant: Optional[str] = "base",
#         model_name: Optional[str] = None,
#         proj_dim: Optional[int] = None,
#         freeze: bool = True,
#         include_cls: bool = True,
#         norm_tokens: bool = False,
#         use_fp16: bool = False,
#     ):
#         super().__init__()
#         if variant is not None:
#             if variant not in SIGLIP2_PRESETS:
#                 raise ValueError(
#                     f"未知 variant='{variant}'。可选：{list(SIGLIP2_PRESETS.keys())}，"
#                     f"或直接传 model_name= 完整权重名。"
#                 )
#             self.model_name = SIGLIP2_PRESETS[variant]
#         elif model_name is not None:
#             self.model_name = model_name
#         else:
#             raise ValueError("必须提供 variant 或 model_name 之一。")

#         self.include_cls = include_cls
#         self.norm_tokens = norm_tokens
#         self.use_fp16 = use_fp16

#         # 加载处理器与模型
#         self.processor = AutoProcessor.from_pretrained(self.model_name)
#         # self.backbone = AutoModel.from_pretrained(self.model_name)

#         local_dir = os.environ.get("SIGLIP2_LOCAL_DIR",
#                            os.path.expanduser("~/hf_cache/models/siglip2-so400m-patch14-384"))
#         cache_dir = os.environ.get("TRANSFORMERS_CACHE",
#                                 os.path.expanduser("~/hf_cache/transformers"))
#         os.makedirs(cache_dir, exist_ok=True)

#         # 尝试优先使用 AutoImageProcessor（它会读 preprocessor_config.json）
#         try:
#             from transformers import AutoImageProcessor as _Proc
#         except Exception:
#             from transformers import AutoProcessor as _Proc  # 兜底

#         if os.path.isdir(local_dir):
#             # 纯本地加载，禁止联网与回退；文件缺失会立刻报错，便于定位
#             self.processor = _Proc.from_pretrained(local_dir, local_files_only=True, cache_dir=cache_dir)
#             self.backbone  = AutoModel.from_pretrained(local_dir, local_files_only=True, cache_dir=cache_dir)
#         else:
#             # 回退到在线，但明确缓存到你可写的目录
#             self.processor = _Proc.from_pretrained(self.model_name, cache_dir=cache_dir)
#             self.backbone  = AutoModel.from_pretrained(self.model_name,  cache_dir=cache_dir)



#         if not hasattr(self.backbone, "vision_model"):
#             raise RuntimeError(
#                 f"{self.model_name} 未暴露 vision_model；请升级 transformers 到支持 SigLIP-2 的版本。"
#             )

#         # 冻结视觉塔
#         if freeze:
#             for p in self.backbone.parameters():
#                 p.requires_grad = False
#             self.backbone.eval()

#         # 获取视觉塔 hidden size
#         hidden_size = getattr(self.backbone.vision_model.config, "hidden_size", None)
#         if hidden_size is None:
#             raise RuntimeError("未在 vision_model.config 找到 hidden_size。")

#         # 可选线性投影
#         if proj_dim is not None and proj_dim != hidden_size:
#             self.proj = nn.Linear(hidden_size, proj_dim)
#             self.out_dim = proj_dim
#         else:
#             self.proj = nn.Identity()
#             self.out_dim = hidden_size

#     @torch.no_grad()
#     def _preprocess_images(
#         self,
#         y: Union[torch.Tensor, List[torch.Tensor]]
#     ) -> dict:
#         """
#         输入:
#           - y: Tensor[B,3,H,W]（float32/float16，建议 0~1），或长度为 B 的 Tensor 列表
#         输出:
#           - dict: processor 打包后的 pixel_values
#         """
#         if isinstance(y, torch.Tensor):
#             assert y.dim() == 4 and y.size(1) == 3, f"期望 y 为 [B,3,H,W]，得到 {tuple(y.shape)}"
#             imgs = []
#             for img in y:
#                 img = img.detach().cpu()
#                 if img.dtype != torch.uint8:
#                     img = img.clamp(0, 1)
#                 pil = to_pil_image(img)
#                 imgs.append(pil)
#         else:
#             imgs = []
#             for img in y:
#                 assert isinstance(img, torch.Tensor) and img.dim() == 3 and img.size(0) == 3
#                 img = img.detach().cpu()
#                 if img.dtype != torch.uint8:
#                     img = img.clamp(0, 1)
#                 pil = to_pil_image(img)
#                 imgs.append(pil)

#         inputs = self.processor(images=imgs, return_tensors="pt")
#         return inputs

#     def list_available_variants(self) -> Dict[str, str]:
#         """返回可选别名到权重名的映射，便于外部打印查看。"""
#         return dict(SIGLIP2_PRESETS)

#     def forward(
#         self,
#         y: torch.Tensor,
#         *,
#         return_attention: bool = False
#     ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
#         """
#         Args:
#             y: [B,3,H,W] RGB 图像批
#             return_attention: 是否返回注意力（若模型支持）

#         Returns:
#             tokens: [B, T, D]，T = token 数（含/不含 CLS 取决于 include_cls），D = out_dim
#             (opt) attn: 视觉塔注意力（若可用且请求）
#         """
#         device = next(self.parameters()).device
#         inputs = self._preprocess_images(y)
#         pixel_values = inputs["pixel_values"].to(device)

#         # autocast 仅在冻结/推理场景下建议开启，节省显存与带宽
#         amp_ctx = torch.autocast(device_type=str(device).split(":")[0], dtype=torch.float16) if self.use_fp16 else torch.cpu.amp.autocast(enabled=False)
#         with amp_ctx:
#             vision_outputs = self.backbone.vision_model(
#                 pixel_values=pixel_values,
#                 output_hidden_states=True,
#                 output_attentions=return_attention,
#             )

#         feats = vision_outputs.last_hidden_state  # [B, N, C], N=1(cls)+num_patches
#         if not self.include_cls:
#             feats = feats[:, 1:, :]  # 去掉 CLS，只保留 patch token

#         if self.norm_tokens:
#             feats = torch.nn.functional.normalize(feats, dim=-1, p=2)

#         tokens = self.proj(feats)  # [B, T, D]

#         if return_attention and hasattr(vision_outputs, "attentions"):
#             return tokens, vision_outputs.attentions
#         return tokens





import os
from typing import Optional, Tuple, Union, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

from transformers import AutoModel

# ===== 预设别名 =====
SIGLIP2_PRESETS: Dict[str, str] = {
    "base-224":    "google/siglip2-base-patch16-224",
    "base-384":    "google/siglip2-base-patch16-384",
    "large-224":   "google/siglip2-large-patch16-224",
    "large-384":   "google/siglip2-large-patch16-384",
    "so400m-384":  "google/siglip2-so400m-patch14-384",
    "so400m-448":  "google/siglip2-so400m-patch14-448",
    "giant-384":   "google/siglip2-gpatch14-384",
    "giant-448":   "google/siglip2-gpatch14-448",
    # 简写
    "base":        "google/siglip2-base-patch16-224",
    "large":       "google/siglip2-large-patch16-384",
    "so400m":      "google/siglip2-so400m-patch14-384",
    "g":           "google/siglip2-gpatch14-448",
}


def _get_image_processor():
    # 优先使用 AutoImageProcessor（读取 preprocessor_config.json）
    try:
        from transformers import AutoImageProcessor as _Proc
    except Exception:
        from transformers import AutoProcessor as _Proc
    return _Proc


def _autocast_ctx(enabled: bool, device: torch.device):
    if not enabled:
        return torch.autocast(device_type="cpu", dtype=torch.float32, enabled=False)  # no-op
    if device.type == "cuda":
        # CUDA上优先 fp16（也可根据需要换成 bf16）
        return torch.cuda.amp.autocast(dtype=torch.float16)
    # CPU上尽量用 bf16（如果硬件/内核支持）
    return torch.autocast(device_type="cpu", dtype=torch.bfloat16)


class SigLIP2TokenExtractor(nn.Module):
    """
    将 RGB 图像批 y: [B,3,H,W] 输入 SigLIP-2 视觉塔，输出一串 patch 特征 token（可选含 CLS），
    并可线性投影到统一维度以便作为下游 Transformer 的额外输入。

    版本选择（优先级从高到低）：
      1) local_dir: 本地完整权重目录（含 config.json / model.safetensors / preprocessor_config.json 等）
      2) model_name: HF 完整权重名（如 'google/siglip2-base-patch16-224'）
      3) variant: 预设别名（见 SIGLIP2_PRESETS）

    Args:
        variant:         预设别名（默认 "base"）。当 local_dir / model_name 均为空时生效。
        model_name:      完整的 Hugging Face 权重名。
        local_dir:       本地模型目录（存在时优先使用，且可与 local_files_only=True 配合离线运行）。
        cache_dir:       HF 缓存目录（默认取环境变量 TRANSFORMERS_CACHE 或 ~/hf_cache/transformers）。
        local_files_only:仅使用本地文件（不联网不回退），默认 False。
        proj_dim:        输出 token 的维度；None 则保持视觉塔 hidden_size。
        freeze:          是否冻结 SigLIP-2 权重（默认 True）。
        include_cls:     返回序列是否包含 CLS token（默认 True）。
        norm_tokens:     是否逐 token 做 L2 归一化（默认 False）。
        use_amp:         使用 autocast（CUDA: fp16；CPU: bf16），默认 False。
    """

    def __init__(
        self,
        *,
        variant: Optional[str] = "base",
        model_name: Optional[str] = None,
        local_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        proj_dim: Optional[int] = None,
        freeze: bool = True,
        include_cls: bool = True,
        norm_tokens: bool = False,
        use_amp: bool = False,
    ):
        super().__init__()

        # -------- 解析目标模型标识 --------
        if local_dir:
            self.load_from = ("local", local_dir)
        elif model_name:
            self.load_from = ("name", model_name)
        else:
            if variant not in SIGLIP2_PRESETS:
                raise ValueError(f"未知 variant='{variant}'；可选：{list(SIGLIP2_PRESETS.keys())}")
            self.load_from = ("name", SIGLIP2_PRESETS[variant])

        # -------- 缓存目录 --------
        if cache_dir is None:
            cache_dir = os.environ.get("TRANSFORMERS_CACHE", os.path.expanduser("~/hf_cache/transformers"))
        os.makedirs(cache_dir, exist_ok=True)

        # -------- 选择 Processor 类 --------
        ProcCls = _get_image_processor()

        # -------- 加载 processor & backbone（统一本地/在线逻辑）--------
        load_kind, path = self.load_from
        if load_kind == "local":
            if not os.path.isdir(path):
                raise FileNotFoundError(f"local_dir 不存在或不可读：{path}")
            self.processor = ProcCls.from_pretrained(path, local_files_only=True, cache_dir=cache_dir)
            self.backbone  = AutoModel.from_pretrained(path,  local_files_only=True, cache_dir=cache_dir)
            self.model_id_for_log = path
        else:
            self.processor = ProcCls.from_pretrained(path, cache_dir=cache_dir, local_files_only=local_files_only)
            self.backbone  = AutoModel.from_pretrained(path,  cache_dir=cache_dir, local_files_only=local_files_only)
            self.model_id_for_log = path

        # -------- 验证视觉塔可用 --------
        vision = getattr(self.backbone, "vision_model", None) or getattr(self.backbone, "vision_tower", None)
        if vision is None:
            raise RuntimeError(f"{self.model_id_for_log} 未暴露 vision_model/vision_tower；请升级 transformers。")
        self.vision = vision  # 仅持有视觉塔

        # -------- 冻结与 eval --------
        if freeze:
            for p in self.vision.parameters():
                p.requires_grad = False
            self.vision.eval()

        # -------- 投影层配置 --------
        hidden_size = getattr(self.vision.config, "hidden_size", None)
        if hidden_size is None:
            raise RuntimeError("未在 vision.config 找到 hidden_size。")

        if proj_dim is not None and proj_dim != hidden_size:
            self.proj = nn.Linear(hidden_size, proj_dim, bias=False)
            self.out_dim = proj_dim
        else:
            self.proj = nn.Identity()
            self.out_dim = hidden_size

        # 其它配置
        self.include_cls = include_cls
        self.norm_tokens = norm_tokens
        self.use_amp = use_amp

    # ---- 工具方法 ----
    @staticmethod
    def _as_pils(batch: Union[torch.Tensor, List[torch.Tensor]]) -> List["PIL.Image.Image"]:
        """
        将 [B,3,H,W] Tensor 或 List[Tensor(3,H,W)] 转成 PIL 列表；假设输入在 0~1 或 uint8。
        """
        pils: List["PIL.Image.Image"] = []
        if isinstance(batch, torch.Tensor):
            assert batch.dim() == 4 and batch.size(1) == 3, f"期望 [B,3,H,W]，得到 {tuple(batch.shape)}"
            for img in batch:
                img = img.detach().cpu()
                if img.dtype != torch.uint8:
                    img = img.clamp(0, 1)
                pils.append(to_pil_image(img))
        else:
            for img in batch:
                assert isinstance(img, torch.Tensor) and img.dim() == 3 and img.size(0) == 3
                img = img.detach().cpu()
                if img.dtype != torch.uint8:
                    img = img.clamp(0, 1)
                pils.append(to_pil_image(img))
        return pils

    def _preprocess_images(self, y: Union[torch.Tensor, List[torch.Tensor]]) -> Dict[str, torch.Tensor]:
        imgs = self._as_pils(y)
        inputs = self.processor(images=imgs, return_tensors="pt")
        return inputs  # {"pixel_values": [B,3,h,w], ...}

    def list_available_variants(self) -> Dict[str, str]:
        return dict(SIGLIP2_PRESETS)

    @torch.no_grad()
    def extract(
        self,
        y: Union[torch.Tensor, List[torch.Tensor]],
        *,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        推理接口（不记录梯度）。返回:
          tokens: [B, T, D]  (T 包含/不含 CLS 取决于 include_cls；D 为 out_dim)
          (opt) attn: 视觉塔注意力（若请求且模型支持）
        """
        device = next(self.parameters()).device
        inputs = self._preprocess_images(y)
        pixel_values = inputs["pixel_values"].to(device, non_blocking=True)

        with _autocast_ctx(self.use_amp, device):
            outputs = self.vision(
                pixel_values=pixel_values,
                output_hidden_states=True,
                output_attentions=return_attention,
            )
            feats = outputs.last_hidden_state  # [B, 1+N, C]

        if not self.include_cls:
            feats = feats[:, 1:, :]  # 去掉 CLS

        if self.norm_tokens:
            feats = F.normalize(feats, dim=-1)

        tokens = self.proj(feats)  # [B, T, D]

        if return_attention and hasattr(outputs, "attentions"):
            return tokens, outputs.attentions
        return tokens

    def forward(
        self,
        y: Union[torch.Tensor, List[torch.Tensor]],
        *,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        # 默认 forward = 训练/微调场景；如需纯推理，请用 extract()
        device = next(self.parameters()).device
        inputs = self._preprocess_images(y)
        pixel_values = inputs["pixel_values"].to(device, non_blocking=True)

        with _autocast_ctx(self.use_amp, device):
            outputs = self.vision(
                pixel_values=pixel_values,
                output_hidden_states=True,
                output_attentions=return_attention,
            )
            feats = outputs.last_hidden_state

        if not self.include_cls:
            feats = feats[:, 1:, :]

        if self.norm_tokens:
            feats = F.normalize(feats, dim=-1)

        tokens = self.proj(feats)

        if return_attention and hasattr(outputs, "attentions"):
            return tokens, outputs.attentions
        return tokens

    def count_params(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
