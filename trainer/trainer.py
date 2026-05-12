# Importing Libraries
import math
import os

import torch
import torchvision
from aim import Run
from utils import reproducibility

from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

from aim import Image, Run
from PIL import Image
import torch.nn as nn

import time

import cv2
import matplotlib.cm as cm

import scipy.io


# 替换原来的 cv2.applyColorMap 部分
def apply_colormap(img_gray, cmap_name="jet"):
    """
    输入: img_gray [H, W] uint8
    输出: [H, W, 3] RGB uint8
    """
    cmap = cm.get_cmap(cmap_name)
    img_color = cmap(img_gray / 255.0)[:, :, :3]  # 归一化到 [0,1]，只取RGB
    img_color = (img_color * 255).astype(np.uint8)
    return img_color

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def _apply_colormap_u8(gray_u8: np.ndarray, cmap_name: str = "jet") -> np.ndarray:
        """
        输入: uint8 灰度 [H, W] (0~255)
        输出: uint8 伪彩 [H, W, 3] (0~255)
        """
        assert gray_u8.dtype == np.uint8 and gray_u8.ndim == 2
        cmap = cm.get_cmap(cmap_name)
        colored = cmap(gray_u8.astype(np.float32) / 255.0)[..., :3]  # [H,W,3], float in [0,1]
        return (colored * 255.0).astype(np.uint8)

def _apply_colormap_u8_from_float01(x01: np.ndarray, cmap_name: str) -> np.ndarray:
    """
    x01: [H,W] float in [0,1]
    return: uint8 [H,W,3]
    """
    x01 = np.clip(x01, 0.0, 1.0).astype(np.float32)
    cmap = cm.get_cmap(cmap_name)
    rgb = cmap(x01)[..., :3]  # [H,W,3], float in [0,1]
    return (rgb * 255.0 + 0.5).astype(np.uint8)




class NMSELoss:
    def __init__(self):
        pass

    def __call__(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        pred = predictions.detach().cpu().numpy()
        targ = targets.detach().cpu().numpy()
        mse_loss = np.mean((pred - targ) ** 2)  # 计算均方误差
        target_variance = np.mean(targ ** 2)  # 计算真实值的均方
        nmse = mse_loss / target_variance  # 计算归一化均方误差
        return nmse




class Trainer_multipath:
    def __init__(
        self,
        vqgan: torch.nn.Module,
        vqgan1: torch.nn.Module,
        vqgan2: torch.nn.Module,
        vqgan3: torch.nn.Module,
        vqgan4: torch.nn.Module,
        transformer: torch.nn.Module,
        run: Run,
        config: dict,
        experiment_dir: str = "experiments/250312",
        seed: int = 42,
        device: str = "cuda",
        multipath_name1: str = "power",
        multipath_name2: str = "power",
        multipath_name3: str = "power",
        multipath_name4: str = "power",
    ) -> None:

        self.vqgan = vqgan
        self.vqgan1 = vqgan1
        self.vqgan2 = vqgan2
        self.vqgan3 = vqgan3
        self.vqgan4 = vqgan4

        self.transformer = transformer
        self.multipath_name1 = multipath_name1
        self.multipath_name2 = multipath_name2
        self.multipath_name3 = multipath_name3
        self.multipath_name4 = multipath_name4

        self.run = run
        self.config = config
        self.experiment_dir = experiment_dir
        self.seed = seed
        self.device = device

        print(f"[INFO] Setting seed to {seed}")
        reproducibility(seed)

        print(f"[INFO] Results will be saved in {experiment_dir}")
        self.experiment_dir = experiment_dir

    

    


    def generate_multipath_nonAR(
        self,
        dataloader: torch.utils.data.DataLoader,
        n_images: int = 100,
        latent_channels=1024,
        upsample_scale: int = 4,          # NEW: 输出图整体上采样倍数（整数）
        upsample_mode: str = "nearest",   # NEW: "nearest" 更清晰；也可以 "bilinear"
    ):
        print(f"[INFO] Generating {n_images} images...")

        self.vqgan.to(self.device)
        self.vqgan1.to(self.device)
        self.vqgan2.to(self.device)
        self.vqgan3.to(self.device)
        self.vqgan4.to(self.device)
        self.transformer = self.transformer.to(self.device)

        save_dir = os.path.join(self.experiment_dir, "generation")
        os.makedirs(save_dir, exist_ok=True)  # 若不存在，则创建

        non_mlp_param_count = 0
        for block in self.transformer.transformer.blocks:
            # ln1、ln2
            for layernorm in [block.ln1, block.ln2]:
                non_mlp_param_count += sum(p.numel() for p in layernorm.parameters())
            # 注意力层 attn
            non_mlp_param_count += sum(p.numel() for p in block.attn.parameters())
            
        i = 0
        criterion = NMSELoss()
        Loss_list = [0.0, 0.0, 0.0, 0.0]  # 分别统计 4 个任务的 Loss
        for index, imgs in enumerate(dataloader):

            RGB = Variable(imgs[0]).to(device=self.device)
            multipath1 = Variable(imgs[1]).to(device=self.device)
            multipath2 = Variable(imgs[2]).to(device=self.device)
            multipath3 = Variable(imgs[3]).to(device=self.device)
            multipath4 = Variable(imgs[4]).to(device=self.device)
            if len(imgs) > 5:
                frequency = Variable(imgs[5]).to(device=self.device)
            else:
                frequency = None
            multipath = [multipath1, multipath2, multipath3, multipath4]

            # --- 新增：path id ---
            if len(imgs) > 6:
                path_id = Variable(imgs[6]).to(device=self.device)
            else:
                path_id = None

            # 1. 编码RGB为token
            _, rgb_indices = self.transformer.encode_to_z(RGB)  # [B, L_rgb]

            # 3. 生成 token
            steps = 64

            # 兼容：sample 可能还不支持 path 参数
            try:
                cir_indices_list = self.transformer.sample(RGB, rgb_indices, steps=steps, freq=frequency, path=path_id)
            except TypeError:
                cir_indices_list = self.transformer.sample(RGB, rgb_indices, steps=steps, freq=frequency)

            # 或者统一成一个列表：
            decode_fns = [
                self.transformer.z_to_image1,
                self.transformer.z_to_image2,
                self.transformer.z_to_image3,
                self.transformer.z_to_image4,
            ]

            # 对每个任务独立解码
            sampled_imgs_list = []

            for j, (indices, decode_fn) in enumerate(zip(cir_indices_list, decode_fns)):
                pred = decode_fn(indices, latent_channels=latent_channels)

                gt = multipath[j]
                if j == 0:
                    pred0 = (pred + 1) / 2 - 1.21
                    gt = (gt + 1) / 2 - 1.21
                elif j == 1:
                    pred0 = (pred + 1) / 2 - 3.5
                    gt = (gt + 1) / 2 - 3.5
                else:
                    pred0 = (pred + 1) / 2
                    gt = (gt + 1) / 2

                loss = criterion(pred0, gt)
                Loss_list[j] += loss.item()  # 累加
                sampled_imgs_list.append(pred)

            sampled_imgs = torch.cat(sampled_imgs_list, dim=1)
            CIR = torch.cat(multipath, dim=1)

            # ================== 可视化 =====================

            task_names = ["delay", "angle", "power", "aoa"]

            for b in range(RGB.shape[0]):  # 遍历 batch 中的每个样本
                images = []

                # ---- 1. 原始 RGB 图像 ----
                rgb_img = RGB[b].detach().cpu()  # (3, H, W)
                rgb_img = (rgb_img + 1) / 2  # Normalize to [0,1]
                rgb_img = rgb_img.clamp(0, 1).numpy()
                rgb_img = np.transpose(rgb_img, (1, 2, 0))  # (H, W, 3)
                rgb_img = (rgb_img * 255).astype(np.uint8)
                images.append(Image.fromarray(rgb_img))

                for ch in range(4):
                    # GT
                    gt_ch = CIR[b, ch].detach().cpu().numpy()  # (H, W)
                    gt_ch = ((gt_ch + 1) / 2.0 * 255).clip(0, 255).astype(np.uint8)  # 映射 [-1,1] → [0,255]
                    gt_color = apply_colormap(gt_ch, cmap_name="jet")
                    images.append(Image.fromarray(gt_color))

                    # Pred
                    pred_ch = sampled_imgs[b, ch].detach().cpu().numpy()  # (H, W)
                    pred_ch = ((pred_ch + 1) / 2.0 * 255).clip(0, 255).astype(np.uint8)  # 映射 [-1,1] → [0,255]
                    pred_color = apply_colormap(pred_ch, cmap_name="jet")
                    images.append(Image.fromarray(pred_color))

                # ---- 拼接九张图 ----
                merged_image = np.concatenate([np.array(img) for img in images], axis=1)  # 横向拼接
                final_image = Image.fromarray(merged_image)

                # ===== NEW: 上采样输出（让图更清晰）=====
                if isinstance(upsample_scale, int) and upsample_scale > 1:
                    w, h = final_image.size
                    if str(upsample_mode).lower() == "bilinear":
                        resample = Image.BILINEAR
                    else:
                        resample = Image.NEAREST
                    final_image = final_image.resize((w * upsample_scale, h * upsample_scale), resample=resample)

                # ---- 保存 ----
                final_image.save(os.path.join(self.experiment_dir, "generation", f"generated_{i}.jpg"))

            i += 1
            if i == n_images:
                break

        for j, l in enumerate(Loss_list):
            print(f"{l / i:.6f}")

