# Importing Libraries
import torch

from dataloader import RGB2pddodDataset_single

import os
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from torch.utils.data import random_split

class TaggedDataset(Dataset):
    def __init__(self, base_dataset, tag):
        super().__init__()
        self.base_dataset = base_dataset
        self.tag = tag  # 通信频点，如 28 或 60

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data = self.base_dataset[idx]  # 可能是 tuple
        return data + (self.tag,)      # 正确地拼接 tuple + tuple

class TaggedDataset2(Dataset):
    def __init__(self, base_dataset, tag, tag2):
        super().__init__()
        self.base_dataset = base_dataset
        self.tag = tag  # 通信频点，如 28 或 60
        self.tag2 = tag2

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data = self.base_dataset[idx]  # 可能是 tuple
        return data + (self.tag,self.tag2)      # 正确地拼接 tuple + tuple


def load_dataloader(
    name: str = "mnist",
    batch_size: int = 8,
    image_size: int = 256,
    num_workers: int = 4,
    save_path: str = "data",
    num_files: int = 10,
    split_dataset = False,  # 默认为 False 不划分
    train_ratio = 0.9  # 默认8:2划分
) -> torch.utils.data.DataLoader:
    """Load the data loader for the given name.

    Args:
        name (str, optional): The name of the data loader. Defaults to "mnist".
        batch_size (int, optional): The batch size. Defaults to 2.
        image_size (int, optional): The image size. Defaults to 256.
        num_workers (int, optional): The number of workers to use for the dataloader. Defaults to 4.
        save_path (str, optional): The path to save the data to. Defaults to "data".

    Returns:
        torch.utils.data.DataLoader: The data loader.
    """

    if name == "mnist":
        return load_mnist(
            batch_size=batch_size,
            image_size=image_size,
            num_workers=num_workers,
            save_path=save_path,
        )

       
    elif name == "RGB2pddod_single_multi_frequency":
        """加载多个对应场景的多模态数据，并添加频点标签"""

        path_groups = [
            {
                "freq": 28.,  # GHz
                "rgb": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/RGB_50",
                "power": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/power/power_crossroad_50",
                "delay": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/delay/delay_crossroad_50",
                "dodphi": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/dod_phi/dod_phi_crossroad_50",
                "dodtheta": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/dod_theta/dod_theta_crossroad_50",
            },
            {
                "freq": 28.,
                "rgb": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/RGB_70",
                "power": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/power/power_crossroad_70",
                "delay": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/delay/delay_crossroad_70",
                "dodphi": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/dod_phi/dod_phi_crossroad_70",
                "dodtheta": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/dod_theta/dod_theta_crossroad_70",
            },
            {
                "freq": 28.,
                "rgb": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/RGB_80",
                "power": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/power/power_crossroad_80",
                "delay": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/delay/delay_crossroad_80",
                "dodphi": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/dod_phi/dod_phi_crossroad_80",
                "dodtheta": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/dod_theta/dod_theta_crossroad_80",
            },
            {
                "freq": 1.6,
                "rgb": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/RGB_70",
                "power": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/power/power_crossroad_70_1.6GHz",
                "delay": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/delay/delay_crossroad_70_1.6GHz",
                "dodphi": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/dod_phi/dod_phi_crossroad_70_1.6GHz",
                "dodtheta": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/dod_theta/dod_theta_crossroad_70_1.6GHz",
            },
            {
                "freq": 5.9,
                "rgb": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/RGB_70",
                "power": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/power/power_crossroad_70_5.9GHz",
                "delay": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/delay/delay_crossroad_70_5.9GHz",
                "dodphi": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/dod_phi/dod_phi_crossroad_70_5.9GHz",
                "dodtheta": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/dod_theta/dod_theta_crossroad_70_5.9GHz",
            },
            {
                "freq": 15.,
                "rgb": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/RGB_70",
                "power": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/power/power_crossroad_70_15GHz",
                "delay": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/delay/delay_crossroad_70_15GHz",
                "dodphi": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/dod_phi/dod_phi_crossroad_70_15GHz",
                "dodtheta": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/dod_theta/dod_theta_crossroad_70_15GHz",
            },
            {
                "freq": 28.,  # GHz
                "rgb": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/RGB_widelane/RGB_widelane_200",
                "power": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/power/power_widelane_200_28",
                "delay": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/delay/delay_widelane_200_28",
                "dodphi": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/dod_phi/dod_phi_widelane_200_28",
                "dodtheta": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/dod_theta/dod_theta_widelane_200_28",
            },
            {
                "freq": 28.,
                "rgb": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/RGB_widelane/RGB_widelane_250",
                "power": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/power/power_widelane_250_28",
                "delay": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/delay/delay_widelane_250_28",
                "dodphi": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/dod_phi/dod_phi_widelane_250_28",
                "dodtheta": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/dod_theta/dod_theta_widelane_250_28",
            },
            {
                "freq": 28.,
                "rgb": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/RGB_widelane/RGB_widelane_300",
                "power": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/power/power_widelane_300_28",
                "delay": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/delay/delay_widelane_300_28",
                "dodphi": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/dod_phi/dod_phi_widelane_300_28",
                "dodtheta": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/dod_theta/dod_theta_widelane_300_28",
            },
            {
                "freq": 1.6,
                "rgb": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/RGB_widelane/RGB_widelane_250",
                "power": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/power/power_widelane_250_1_6",
                "delay": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/delay/delay_widelane_250_1_6",
                "dodphi": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/dod_phi/dod_phi_widelane_250_1_6",
                "dodtheta": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/dod_theta/dod_theta_widelane_250_1_6",
            },
            {
                "freq": 5.9,
                "rgb": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/RGB_widelane/RGB_widelane_250",
                "power": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/power/power_widelane_250_5_9",
                "delay": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/delay/delay_widelane_250_5_9",
                "dodphi": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/dod_phi/dod_phi_widelane_250_5_9",
                "dodtheta": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/dod_theta/dod_theta_widelane_250_5_9",
            },
            {
                "freq": 15.,
                "rgb": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/RGB_widelane/RGB_widelane_250",
                "power": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/power/power_widelane_250_15",
                "delay": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/delay/delay_widelane_250_15",
                "dodphi": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/dod_phi/dod_phi_widelane_250_15",
                "dodtheta": "/data1/PCNI2_data/hzr/DDPM/DDPM-main/data/dod_theta/dod_theta_widelane_250_15",
            },
        ]

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        transform2 = transforms.Compose([
            transforms.Resize((image_size, image_size)),
        ])

        num_files_list = [num_files, num_files, num_files, num_files, num_files, num_files, 2090, 2090, 2090, 2090, 2090, 2090]  # 每个路径对应的文件数限制

        datasets = []
        for group, nf in zip(path_groups, num_files_list):
            base_dataset = RGB2pddodDataset_single(
                data_dir=group["rgb"],
                data_dir1=group["power"],
                data_dir2=group["delay"],
                data_dir3=group["dodphi"],
                data_dir4=group["dodtheta"],
                transform=transform,
                transform2=transform2,
                num_files=nf
            )
            tagged_dataset = TaggedDataset(base_dataset, group["freq"])
            datasets.append(tagged_dataset)

        combined_dataset = torch.utils.data.ConcatDataset(datasets)

        if split_dataset:
            generator = torch.Generator().manual_seed(42)  # 你可以设置任意整数种子
            # 按照比例拆分训练集和测试集
            total_len = len(combined_dataset)
            train_len = int(train_ratio * total_len)
            test_len = total_len - train_len
            train_dataset, test_dataset = random_split(combined_dataset, [train_len, test_len], generator=generator)

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            return train_loader, test_loader  
        
        else:
            # 不划分，只返回全部数据
            dataloader = DataLoader(
                combined_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
            )
            return dataloader
          

