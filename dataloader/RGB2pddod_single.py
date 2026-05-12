import os
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import scipy.io
import numpy as np
import math
# import h5py

def normalize_dbm(power_dbm, min_dbm=-200, max_dbm=-35):
    """
    将dBm单位的功率归一化到[-1,1]范围
    Args:
        power_dbm: dBm单位的功率值
        min_dbm: 最小功率值，默认-120dBm
        max_dbm: 最大功率值，默认0dBm
    Returns:
        normalized: 归一化到[-1,1]范围的功率值
    """
    # 将零值替换为最小功率值
    zero_mask = (power_dbm == 0)
    power_dbm = torch.where(zero_mask, torch.tensor(min_dbm, device=power_dbm.device), power_dbm)
    
    # 线性归一化到[-1,1]
    normalized = 2 * (power_dbm - min_dbm) / (max_dbm - min_dbm) - 1
    # 限制在[-1,1]范围内
    normalized = torch.clamp(normalized, -1, 1)
    return normalized

def normalize_delay(delay_ns, min_delay=1.6e-7, max_delay=1.7e-5, eps=1e-10):
    """
    将时延归一化到[-1,1]范围，并将零值替换为最大时延值
    Args:
        delay_ns: 纳秒单位的时延值
        min_delay: 最小时延值，默认0ns
        max_delay: 最大时延值，默认1000ns
    Returns:
        normalized: 归一化到[-1,1]范围的时延值
    """
    # 将零值替换为最大时延值
    zero_mask = (delay_ns == 0)
    delay_ns = torch.where(zero_mask, torch.tensor(max_delay, device=delay_ns.device), delay_ns)
    
    # 取对数变换（以10为底）
    delay_log = torch.log10(delay_ns + eps)
    
    # 计算对数域的最大最小值
    min_delay_log = torch.log10(torch.tensor(min_delay) + eps)
    max_delay_log = torch.log10(torch.tensor(max_delay) + eps)
    
    # 在对数域进行线性归一化到[-1,1]
    normalized = 2 * (delay_log - min_delay_log) / (max_delay_log - min_delay_log) - 1
    
    # 限制在[-1,1]范围内
    normalized = torch.clamp(normalized, -1, 1)
    
    return normalized

def numeric_sort(file_name):
    # 提取文件名中的数字部分并转换为整数
    return int(''.join(filter(str.isdigit, file_name)))

class RGB2pddodDataset_single(Dataset):
    """A custom dataset for loading images from a single folder (without categories)."""

    def __init__(self, data_dir, data_dir1, data_dir2, data_dir3, data_dir4, transform=None, transform2=None, num_files=10):
        self.data_dir = data_dir
        self.transform = transform
        # self.image_paths = sorted([os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if
        #                            fname.endswith(('.png', '.jpg', '.jpeg'))], key=numeric_sort)

        self.image_files = [f'image{i}.png' for i in range(1, num_files + 1)]
        self.image_files2 = [f'time{i}_RGB.png' for i in range(1, num_files + 1)]

        self.power_files = [f'time{i}_pd_data_path1_power.mat' for i in range(1, num_files + 1)]
        self.delay_files = [f'time{i}_pd_data_path1_delay.mat' for i in range(1, num_files + 1)]
        self.dod_phi_files = [f'time{i}_dod_data_path1_phi.mat' for i in range(1, num_files + 1)]
        self.dod_theta_files = [f'time{i}_dod_data_path1_theta.mat' for i in range(1, num_files + 1)]

        self.power_path = data_dir1
        self.delay_path = data_dir2
        self.dod_phi_path = data_dir3
        self.dod_theta_path = data_dir4

        self.transform2 = transform2

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        # image_path = os.path.join(self.data_dir, self.image_files[idx])

        try:
            image_path = os.path.join(self.data_dir, self.image_files[idx])
            image = Image.open(image_path).convert("RGB")  # Open image and convert to RGB
        except (FileNotFoundError, OSError):
            # 如果失败，尝试第二种规则：time{i}_RGB.png
            image_path = os.path.join(self.data_dir, self.image_files2[idx])
            image = Image.open(image_path).convert("RGB")

        # image = Image.open(image_path).convert("RGB")  # Open image and convert to RGB
        
        image = image.rotate(270, expand=True)
        # if 0 <= idx <= 1050-1:
        #     image = image.rotate(270, expand=True)
        # else:
        #     image = image.rotate(0, expand=True)

        # print(type(image)) #<class 'PIL.Image.Image'>

        if self.transform:
            image = self.transform(image)  # Apply transformations (e.g., resize, normalize)

        # print(type(image))  # <class 'PIL.Image.Image'>

        #--------------------------------------------power----------------------------------------------------- 
        # 读取.mat文件
        mat_file = os.path.join(self.power_path, self.power_files[idx])
        data = scipy.io.loadmat(mat_file)
        # try:
        #     # 尝试用 scipy 读取（适用于 v7.2 及以下）
        #     data = scipy.io.loadmat(mat_file)
        # except NotImplementedError:
        #     # 如果是 v7.3（HDF5 格式），改用 h5py
        #     with h5py.File(mat_file, 'r') as f:
        #         data = {k: np.array(f[k]) for k in f.keys()}

        # 根据文件名生成变量名
        input_time_var = f'pd_data'  # 例如input_time1, input_time2等
        input_time = data[input_time_var]  # shape (1, 3362)

        power0 = np.float32(input_time)

        power = np.fliplr(power0).copy()

        power = torch.tensor(power)
        power = power.unsqueeze(0)

        if self.transform2:
            power = self.transform2(power)  # Apply transformations (e.g., resize, normalize)
            power = normalize_dbm(power)  # 归一化dBm值

        #--------------------------------------------delay----------------------------------------------------- 
        # 读取.mat文件
        mat_file = os.path.join(self.delay_path, self.delay_files[idx])
        data = scipy.io.loadmat(mat_file)
        # try:
        #     # 尝试用 scipy 读取（适用于 v7.2 及以下）
        #     data = scipy.io.loadmat(mat_file)
        # except NotImplementedError:
        #     # 如果是 v7.3（HDF5 格式），改用 h5py
        #     with h5py.File(mat_file, 'r') as f:
        #         data = {k: np.array(f[k]) for k in f.keys()}

        # 根据文件名生成变量名
        input_time_var = f'pd_data'  # 例如input_time1, input_time2等
        input_time = data[input_time_var]  # shape (1, 3362)

        delay0 = np.float32(input_time)

        # DoA = np.array(DoA).reshape(81, 81)
        delay = np.fliplr(delay0).copy()

        delay = torch.tensor(delay)
        delay = delay.unsqueeze(0)

        if self.transform2:
            delay = self.transform2(delay)  # Apply transformations (e.g., resize, normalize)
            delay = normalize_delay(delay)
            

        #--------------------------------------------dod_phi----------------------------------------------------- 
        # 读取.mat文件
        mat_file = os.path.join(self.dod_phi_path, self.dod_phi_files[idx])
        data = scipy.io.loadmat(mat_file)
        # try:
        #     # 尝试用 scipy 读取（适用于 v7.2 及以下）
        #     data = scipy.io.loadmat(mat_file)
        # except NotImplementedError:
        #     # 如果是 v7.3（HDF5 格式），改用 h5py
        #     with h5py.File(mat_file, 'r') as f:
        #         data = {k: np.array(f[k]) for k in f.keys()}

        # 根据文件名生成变量名
        input_time_var = f'angle_data'  # 例如input_time1, input_time2等
        input_time = data[input_time_var]  # shape (1, 3362)

        dod_phi0 = np.float32(input_time)

        dod_phi = np.fliplr(dod_phi0).copy()

        dod_phi = torch.tensor(dod_phi)
        dod_phi = dod_phi.unsqueeze(0)

        if self.transform2:
            dod_phi = self.transform2(dod_phi)  # Apply transformations (e.g., resize, normalize)
            dod_phi = torch.sin(dod_phi * math.pi / 180.0)  # 角度转弧度并取sin值

        #--------------------------------------------dod_theta----------------------------------------------------- 
        # 读取.mat文件
        mat_file = os.path.join(self.dod_theta_path, self.dod_theta_files[idx])
        data = scipy.io.loadmat(mat_file)
        # try:
        #     # 尝试用 scipy 读取（适用于 v7.2 及以下）
        #     data = scipy.io.loadmat(mat_file)
        # except NotImplementedError:
        #     # 如果是 v7.3（HDF5 格式），改用 h5py
        #     with h5py.File(mat_file, 'r') as f:
        #         data = {k: np.array(f[k]) for k in f.keys()}

        # 根据文件名生成变量名
        input_time_var = f'angle_data'  # 例如input_time1, input_time2等
        input_time = data[input_time_var]  # shape (1, 3362)

        dod_theta0 = np.float32(input_time)

        dod_theta = np.fliplr(dod_theta0).copy()

        dod_theta = torch.tensor(dod_theta)
        dod_theta = dod_theta.unsqueeze(0)

        if self.transform2:
            dod_theta = self.transform2(dod_theta)  # Apply transformations (e.g., resize, normalize)
            dod_theta = torch.sin(dod_theta * math.pi / 180.0)  # 角度转弧度并取sin值


        # combined_tensor = torch.cat([power, delay, dod_phi, dod_theta], dim=0)  # [4, H, W]

        return image, power, delay, dod_phi, dod_theta


class RGB2pdaoaDataset_single(Dataset):
    """A custom dataset for loading images from a single folder (without categories)."""

    def __init__(self, data_dir, data_dir1, data_dir2, data_dir3, data_dir4, transform=None, transform2=None, num_files=10):
        self.data_dir = data_dir
        self.transform = transform
        # self.image_paths = sorted([os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if
        #                            fname.endswith(('.png', '.jpg', '.jpeg'))], key=numeric_sort)

        self.image_files = [f'image{i}.png' for i in range(1, num_files + 1)]
        self.image_files2 = [f'time{i}_RGB.png' for i in range(1, num_files + 1)]

        self.power_files = [f'time{i}_pd_data_path1_power.mat' for i in range(1, num_files + 1)]
        self.delay_files = [f'time{i}_pd_data_path1_delay.mat' for i in range(1, num_files + 1)]
        self.aoa_phi_files = [f'time{i}_aoa_data_path1_phi.mat' for i in range(1, num_files + 1)]
        self.aoa_theta_files = [f'time{i}_aoa_data_path1_theta.mat' for i in range(1, num_files + 1)]

        self.power_path = data_dir1
        self.delay_path = data_dir2
        self.aoa_phi_path = data_dir3
        self.aoa_theta_path = data_dir4

        self.transform2 = transform2

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        # image_path = os.path.join(self.data_dir, self.image_files[idx])

        try:
            image_path = os.path.join(self.data_dir, self.image_files[idx])
            image = Image.open(image_path).convert("RGB")  # Open image and convert to RGB
        except (FileNotFoundError, OSError):
            # 如果失败，尝试第二种规则：time{i}_RGB.png
            image_path = os.path.join(self.data_dir, self.image_files2[idx])
            image = Image.open(image_path).convert("RGB")

        # image = Image.open(image_path).convert("RGB")  # Open image and convert to RGB
        
        image = image.rotate(270, expand=True)
        # if 0 <= idx <= 1050-1:
        #     image = image.rotate(270, expand=True)
        # else:
        #     image = image.rotate(0, expand=True)

        # print(type(image)) #<class 'PIL.Image.Image'>

        if self.transform:
            image = self.transform(image)  # Apply transformations (e.g., resize, normalize)

        # print(type(image))  # <class 'PIL.Image.Image'>

        #--------------------------------------------power----------------------------------------------------- 
        # 读取.mat文件
        mat_file = os.path.join(self.power_path, self.power_files[idx])
        data = scipy.io.loadmat(mat_file)
        # try:
        #     # 尝试用 scipy 读取（适用于 v7.2 及以下）
        #     data = scipy.io.loadmat(mat_file)
        # except NotImplementedError:
        #     # 如果是 v7.3（HDF5 格式），改用 h5py
        #     with h5py.File(mat_file, 'r') as f:
        #         data = {k: np.array(f[k]) for k in f.keys()}

        # 根据文件名生成变量名
        input_time_var = f'pd_data'  # 例如input_time1, input_time2等
        input_time = data[input_time_var]  # shape (1, 3362)

        power0 = np.float32(input_time)

        power = np.fliplr(power0).copy()

        power = torch.tensor(power)
        power = power.unsqueeze(0)

        if self.transform2:
            power = self.transform2(power)  # Apply transformations (e.g., resize, normalize)
            power = normalize_dbm(power)  # 归一化dBm值

        #--------------------------------------------delay----------------------------------------------------- 
        # 读取.mat文件
        mat_file = os.path.join(self.delay_path, self.delay_files[idx])
        data = scipy.io.loadmat(mat_file)
        # try:
        #     # 尝试用 scipy 读取（适用于 v7.2 及以下）
        #     data = scipy.io.loadmat(mat_file)
        # except NotImplementedError:
        #     # 如果是 v7.3（HDF5 格式），改用 h5py
        #     with h5py.File(mat_file, 'r') as f:
        #         data = {k: np.array(f[k]) for k in f.keys()}

        # 根据文件名生成变量名
        input_time_var = f'pd_data'  # 例如input_time1, input_time2等
        input_time = data[input_time_var]  # shape (1, 3362)

        delay0 = np.float32(input_time)

        # DoA = np.array(DoA).reshape(81, 81)
        delay = np.fliplr(delay0).copy()

        delay = torch.tensor(delay)
        delay = delay.unsqueeze(0)

        if self.transform2:
            delay = self.transform2(delay)  # Apply transformations (e.g., resize, normalize)
            delay = normalize_delay(delay)
            

        #--------------------------------------------aoa_phi----------------------------------------------------- 
        # 读取.mat文件
        mat_file = os.path.join(self.aoa_phi_path, self.aoa_phi_files[idx])
        data = scipy.io.loadmat(mat_file)
        # try:
        #     # 尝试用 scipy 读取（适用于 v7.2 及以下）
        #     data = scipy.io.loadmat(mat_file)
        # except NotImplementedError:
        #     # 如果是 v7.3（HDF5 格式），改用 h5py
        #     with h5py.File(mat_file, 'r') as f:
        #         data = {k: np.array(f[k]) for k in f.keys()}

        # 根据文件名生成变量名
        input_time_var = f'angle_data'  # 例如input_time1, input_time2等
        input_time = data[input_time_var]  # shape (1, 3362)

        aoa_phi0 = np.float32(input_time)

        aoa_phi = np.fliplr(aoa_phi0).copy()

        aoa_phi = torch.tensor(aoa_phi)
        aoa_phi = aoa_phi.unsqueeze(0)

        if self.transform2:
            aoa_phi = self.transform2(aoa_phi)  # Apply transformations (e.g., resize, normalize)
            aoa_phi = torch.sin(aoa_phi * math.pi / 180.0)  # 角度转弧度并取sin值

        #--------------------------------------------aoa_theta----------------------------------------------------- 
        # 读取.mat文件
        mat_file = os.path.join(self.aoa_theta_path, self.aoa_theta_files[idx])
        data = scipy.io.loadmat(mat_file)
        # try:
        #     # 尝试用 scipy 读取（适用于 v7.2 及以下）
        #     data = scipy.io.loadmat(mat_file)
        # except NotImplementedError:
        #     # 如果是 v7.3（HDF5 格式），改用 h5py
        #     with h5py.File(mat_file, 'r') as f:
        #         data = {k: np.array(f[k]) for k in f.keys()}

        # 根据文件名生成变量名
        input_time_var = f'angle_data'  # 例如input_time1, input_time2等
        input_time = data[input_time_var]  # shape (1, 3362)

        aoa_theta0 = np.float32(input_time)

        aoa_theta = np.fliplr(aoa_theta0).copy()

        aoa_theta = torch.tensor(aoa_theta)
        aoa_theta = aoa_theta.unsqueeze(0)

        if self.transform2:
            aoa_theta = self.transform2(aoa_theta)  # Apply transformations (e.g., resize, normalize)
            aoa_theta = torch.sin(aoa_theta * math.pi / 180.0)  # 角度转弧度并取sin值


        # combined_tensor = torch.cat([power, delay, aoa_phi, aoa_theta], dim=0)  # [4, H, W]

        return image, power, delay, aoa_phi, aoa_theta
    


class RGB2pddodDataset_multiple_path(Dataset):
    """A custom dataset for loading images from a single folder (without categories)."""

    def __init__(self, data_dir, data_dir1, data_dir2, data_dir3, data_dir4, transform=None, transform2=None, num_files=10):
        self.data_dir = data_dir
        self.transform = transform
        # self.image_paths = sorted([os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if
        #                            fname.endswith(('.png', '.jpg', '.jpeg'))], key=numeric_sort)

        self.image_files = [f'image{i}.png' for i in range(1, num_files + 1)]
        self.image_files2 = [f'time{i}_RGB.png' for i in range(1, num_files + 1)]

        self.power_files = [f'time{i}_pd_data_path1_power.mat' for i in range(1, num_files + 1)]
        self.delay_files = [f'time{i}_pd_data_path1_delay.mat' for i in range(1, num_files + 1)]
        self.dod_phi_files = [f'time{i}_dod_data_path1_phi.mat' for i in range(1, num_files + 1)]
        self.dod_theta_files = [f'time{i}_dod_data_path1_theta.mat' for i in range(1, num_files + 1)]

        self.power_path = data_dir1
        self.delay_path = data_dir2
        self.dod_phi_path = data_dir3
        self.dod_theta_path = data_dir4

        self.transform2 = transform2

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        # image_path = os.path.join(self.data_dir, self.image_files[idx])

        try:
            image_path = os.path.join(self.data_dir, self.image_files[idx])
            image = Image.open(image_path).convert("RGB")  # Open image and convert to RGB
        except (FileNotFoundError, OSError):
            # 如果失败，尝试第二种规则：time{i}_RGB.png
            image_path = os.path.join(self.data_dir, self.image_files2[idx])
            image = Image.open(image_path).convert("RGB")

        # image = Image.open(image_path).convert("RGB")  # Open image and convert to RGB
        
        image = image.rotate(270, expand=True)
        # if 0 <= idx <= 1050-1:
        #     image = image.rotate(270, expand=True)
        # else:
        #     image = image.rotate(0, expand=True)

        # print(type(image)) #<class 'PIL.Image.Image'>

        if self.transform:
            image = self.transform(image)  # Apply transformations (e.g., resize, normalize)

        # print(type(image))  # <class 'PIL.Image.Image'>

        #--------------------------------------------power----------------------------------------------------- 
        # # 读取.mat文件
        # mat_file = os.path.join(self.power_path, self.power_files[idx])

        # 优先使用当前记录的文件名，不存在则尝试 path1..path6
        base = self.power_files[idx]
        # time{i}_pd_data_pathX_power.mat -> (prefix, X, suffix)
        # 也可能你之后会换成别的占位，这里稳妥做法是定位 'path' 片段：
        if '_pd_data_path' in base and '_power.mat' in base:
            prefix = base.split('_pd_data_path')[0] + '_pd_data_path'  # 'time{i}_pd_data_path'
            suffix = '_power.mat'
        else:
            # 若命名不符合预期，就直接按当前名尝试
            candidate_paths = [os.path.join(self.power_path, base)]
        if '_pd_data_path' in base and '_power.mat' in base:
            candidate_paths = [
                os.path.join(self.power_path, f"{prefix}{k}{suffix}") for k in range(1, 7)
            ]

        chosen = None
        tried = []
        for p in candidate_paths:
            tried.append(os.path.basename(p))
            if os.path.exists(p):
                chosen = p
                break

        if chosen is None:
            raise FileNotFoundError(
                f"No file found for index {idx}.\nTried:\n  " + "\n  ".join(tried)
            )

        mat_file = chosen
        data = scipy.io.loadmat(mat_file)
        # try:
        #     # 尝试用 scipy 读取（适用于 v7.2 及以下）
        #     data = scipy.io.loadmat(mat_file)
        # except NotImplementedError:
        #     # 如果是 v7.3（HDF5 格式），改用 h5py
        #     with h5py.File(mat_file, 'r') as f:
        #         data = {k: np.array(f[k]) for k in f.keys()}

        # 根据文件名生成变量名
        input_time_var = f'pd_data'  # 例如input_time1, input_time2等
        input_time = data[input_time_var]  # shape (1, 3362)

        power0 = np.float32(input_time)

        power = np.fliplr(power0).copy()

        power = torch.tensor(power)
        power = power.unsqueeze(0)

        if self.transform2:
            power = self.transform2(power)  # Apply transformations (e.g., resize, normalize)
            power = normalize_dbm(power)  # 归一化dBm值

        #--------------------------------------------delay----------------------------------------------------- 
        # # 读取.mat文件
        # mat_file = os.path.join(self.delay_path, self.delay_files[idx])

        # 优先使用当前记录的文件名，不存在则尝试 path1..path6
        base = self.delay_files[idx]
        # time{i}_pd_data_pathX_delay.mat -> (prefix, X, suffix)
        # 也可能你之后会换成别的占位，这里稳妥做法是定位 'path' 片段：
        if '_pd_data_path' in base and '_delay.mat' in base:
            prefix = base.split('_pd_data_path')[0] + '_pd_data_path'  # 'time{i}_pd_data_path'
            suffix = '_delay.mat'
        else:
            # 若命名不符合预期，就直接按当前名尝试
            candidate_paths = [os.path.join(self.delay_path, base)]
        if '_pd_data_path' in base and '_delay.mat' in base:
            candidate_paths = [
                os.path.join(self.delay_path, f"{prefix}{k}{suffix}") for k in range(1, 7)
            ]

        chosen = None
        tried = []
        for p in candidate_paths:
            tried.append(os.path.basename(p))
            if os.path.exists(p):
                chosen = p
                break

        if chosen is None:
            raise FileNotFoundError(
                f"No file found for index {idx}.\nTried:\n  " + "\n  ".join(tried)
            )

        mat_file = chosen
        data = scipy.io.loadmat(mat_file)
        # try:
        #     # 尝试用 scipy 读取（适用于 v7.2 及以下）
        #     data = scipy.io.loadmat(mat_file)
        # except NotImplementedError:
        #     # 如果是 v7.3（HDF5 格式），改用 h5py
        #     with h5py.File(mat_file, 'r') as f:
        #         data = {k: np.array(f[k]) for k in f.keys()}

        # 根据文件名生成变量名
        input_time_var = f'pd_data'  # 例如input_time1, input_time2等
        input_time = data[input_time_var]  # shape (1, 3362)

        delay0 = np.float32(input_time)

        # DoA = np.array(DoA).reshape(81, 81)
        delay = np.fliplr(delay0).copy()

        delay = torch.tensor(delay)
        delay = delay.unsqueeze(0)

        if self.transform2:
            delay = self.transform2(delay)  # Apply transformations (e.g., resize, normalize)
            delay = normalize_delay(delay)
            

        #--------------------------------------------dod_phi----------------------------------------------------- 
        # # 读取.mat文件
        # mat_file = os.path.join(self.dod_phi_path, self.dod_phi_files[idx])

        # 优先使用当前记录的文件名，不存在则尝试 path1..path6
        base = self.dod_phi_files[idx]
        # time{i}_pd_data_pathX_dodphi.mat -> (prefix, X, suffix)
        # 也可能你之后会换成别的占位，这里稳妥做法是定位 'path' 片段：
        if '_dod_data_path' in base and '_phi.mat' in base:
            prefix = base.split('_dod_data_path')[0] + '_dod_data_path'  # 'time{i}_pd_data_path'
            suffix = '_phi.mat'
        else:
            # 若命名不符合预期，就直接按当前名尝试
            candidate_paths = [os.path.join(self.dod_phi_path, base)]
        if '_dod_data_path' in base and '_phi.mat' in base:
            candidate_paths = [
                os.path.join(self.dod_phi_path, f"{prefix}{k}{suffix}") for k in range(1, 7)
            ]

        chosen = None
        tried = []
        for p in candidate_paths:
            tried.append(os.path.basename(p))
            if os.path.exists(p):
                chosen = p
                break

        if chosen is None:
            raise FileNotFoundError(
                f"No file found for index {idx}.\nTried:\n  " + "\n  ".join(tried)
            )

        mat_file = chosen
        data = scipy.io.loadmat(mat_file)
        # try:
        #     # 尝试用 scipy 读取（适用于 v7.2 及以下）
        #     data = scipy.io.loadmat(mat_file)
        # except NotImplementedError:
        #     # 如果是 v7.3（HDF5 格式），改用 h5py
        #     with h5py.File(mat_file, 'r') as f:
        #         data = {k: np.array(f[k]) for k in f.keys()}

        # 根据文件名生成变量名
        input_time_var = f'angle_data'  # 例如input_time1, input_time2等
        input_time = data[input_time_var]  # shape (1, 3362)

        dod_phi0 = np.float32(input_time)

        dod_phi = np.fliplr(dod_phi0).copy()

        dod_phi = torch.tensor(dod_phi)
        dod_phi = dod_phi.unsqueeze(0)

        if self.transform2:
            dod_phi = self.transform2(dod_phi)  # Apply transformations (e.g., resize, normalize)
            dod_phi = torch.sin(dod_phi * math.pi / 180.0)  # 角度转弧度并取sin值

        #--------------------------------------------dod_theta----------------------------------------------------- 
        # # 读取.mat文件
        # mat_file = os.path.join(self.dod_theta_path, self.dod_theta_files[idx])

        # 优先使用当前记录的文件名，不存在则尝试 path1..path6
        base = self.dod_theta_files[idx]
        # time{i}_pd_data_pathX_dodtheta.mat -> (prefix, X, suffix)
        # 也可能你之后会换成别的占位，这里稳妥做法是定位 'path' 片段：
        if '_dod_data_path' in base and '_theta.mat' in base:
            prefix = base.split('_dod_data_path')[0] + '_dod_data_path'  # 'time{i}_pd_data_path'
            suffix = '_theta.mat'
        else:
            # 若命名不符合预期，就直接按当前名尝试
            candidate_paths = [os.path.join(self.dod_theta_path, base)]
        if '_dod_data_path' in base and '_theta.mat' in base:
            candidate_paths = [
                os.path.join(self.dod_theta_path, f"{prefix}{k}{suffix}") for k in range(1, 7)
            ]

        chosen = None
        tried = []
        for p in candidate_paths:
            tried.append(os.path.basename(p))
            if os.path.exists(p):
                chosen = p
                break

        if chosen is None:
            raise FileNotFoundError(
                f"No file found for index {idx}.\nTried:\n  " + "\n  ".join(tried)
            )

        mat_file = chosen
        data = scipy.io.loadmat(mat_file)
        # try:
        #     # 尝试用 scipy 读取（适用于 v7.2 及以下）
        #     data = scipy.io.loadmat(mat_file)
        # except NotImplementedError:
        #     # 如果是 v7.3（HDF5 格式），改用 h5py
        #     with h5py.File(mat_file, 'r') as f:
        #         data = {k: np.array(f[k]) for k in f.keys()}

        # 根据文件名生成变量名
        input_time_var = f'angle_data'  # 例如input_time1, input_time2等
        input_time = data[input_time_var]  # shape (1, 3362)

        dod_theta0 = np.float32(input_time)

        dod_theta = np.fliplr(dod_theta0).copy()

        dod_theta = torch.tensor(dod_theta)
        dod_theta = dod_theta.unsqueeze(0)

        if self.transform2:
            dod_theta = self.transform2(dod_theta)  # Apply transformations (e.g., resize, normalize)
            dod_theta = torch.sin(dod_theta * math.pi / 180.0)  # 角度转弧度并取sin值


        # combined_tensor = torch.cat([power, delay, dod_phi, dod_theta], dim=0)  # [4, H, W]

        return image, power, delay, dod_phi, dod_theta