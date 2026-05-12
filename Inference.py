import argparse
import os
import yaml
from aim import Run

from dataloader import load_dataloader
# from sentry_sdk.utils import epoch
from trainer import Trainer_multipath
from transformer import multiVQGANTransformer
from vit_vqgan import VQGAN
import torch


# 如果用siglip，需要先执行命令export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"


# 定义主函数，接收命令行参数和配置字典
def main(args, config):
    # 1. 模型初始化部分
    vqgan_RGB = VQGAN(**config["architecture"]["vqgan_RGB"])  # 从配置文件中初始化VQGAN模型
    vqgan_power = VQGAN(**config["architecture"]["vqgan_CIR"])  # 从配置文件中初始化VQGAN模型
    vqgan_delay = VQGAN(**config["architecture"]["vqgan_CIR"])
    vqgan_dodphi = VQGAN(**config["architecture"]["vqgan_CIR"])
    vqgan_dodtheta = VQGAN(**config["architecture"]["vqgan_CIR"])
  
    transformer = multiVQGANTransformer(  # 初始化VQGAN-Transformer联合模型
        vqgan=vqgan_RGB,
        vqgan1=vqgan_power, # 传入已创建的VQGAN实例
        vqgan2=vqgan_delay,
        vqgan3=vqgan_dodphi,
        vqgan4=vqgan_dodtheta,
        **config["architecture"]["transformer"],  # 从配置读取transformer参数
        device=args.device  # 设置运行设备
    )
    print("transformer",transformer)
    
    # 2. 数据加载部分
    num_samples=1830
    
    # 3. 实验跟踪部分
    run = Run(experiment=args.dataset_name)  
    run["hparams"] = config  # 记录超参数到实验跟踪系统

    # 4. 训练器初始化
    experiment_dir= "experiments/260513"

    load_RGB_dir= "experiments/RGB"
    load_multipath_dir1= "experiments/power"
    load_multipath_dir2= "experiments/delay"
    load_multipath_dir3= "experiments/dodphi"
    load_multipath_dir4= "experiments/dodtheta"
    load_transformer= "experiments/transformer"


    
    trainer = Trainer_multipath(  # 创建训练管理对象
        vqgan_RGB, vqgan_power, vqgan_delay, vqgan_dodphi, vqgan_dodtheta, 
        transformer,  # 传入模型
        run=run,  # 绑定实验记录
        config=config["trainer"],  # 读取训练相关配置
        seed=args.seed,  # 设置随机种子
        device=args.device,  # 指定运行设备
        experiment_dir= experiment_dir,
        multipath_name1=args.multipath_name1,  # 指定多路径数据集名称
        multipath_name2=args.multipath_name2,
        multipath_name3=args.multipath_name3,
        multipath_name4=args.multipath_name4,
    )

    
    transformer.load_checkpoint(os.path.join(load_transformer, "checkpoints", "transformer.pt"), device=args.device)
    

    dataloader2, dataloader_generate2 = load_dataloader(name=args.dataset_name, batch_size=1,
                                 image_size=config["architecture"]["vqgan_CIR"]["img_size"], num_files=num_samples, split_dataset=True, train_ratio=0.9)
    trainer.generate_multipath_nonAR(dataloader_generate2,n_images=1500,latent_channels=512)  # 生成最终图像
    



# 主程序入口
if __name__ == "__main__":
    # 6. 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument(  # 添加配置文件路径参数
        "--config_path", type=str,
        default="configs/default_crossmodal_vitvqgan_base.yml",
        help="path to config file"
    )
    parser.add_argument(  # 添加数据集选择参数
        "--dataset_name", type=str,
        choices=["RGB2power", "RGB2dodtheta", "RGB2dodphi", "RGB2delay", "RGB2pddod_single",
                 "RGB2pddod_single_frequency","RGB2pddod_single_multi_frequency","RGB2pdaoa","RGB2pddod_multiple_path"],
        default="RGB2pddod_single_multi_frequency",
        help="Dataset for the model"
    )
    parser.add_argument(  # 添加数据集选择参数
        "--multipath_name1", type=str,
        choices=["mnist", "cifar", "custom", "RGB2CIR", "RGB2power", "RGB2dodtheta", "RGB2dodphi", "RGB2delay"],
        default="power",
        help="multipath Dataset for the model"
    )
    parser.add_argument(  # 添加数据集选择参数
        "--multipath_name2", type=str,
        choices=["mnist", "cifar", "custom", "RGB2CIR", "RGB2power", "RGB2dodtheta", "RGB2dodphi", "RGB2delay"],
        default="delay",
        help="multipath Dataset for the model"
    )
    parser.add_argument(  # 添加数据集选择参数
        "--multipath_name3", type=str,
        choices=["mnist", "cifar", "custom", "RGB2CIR", "RGB2power", "RGB2dodtheta", "RGB2dodphi", "RGB2delay"],
        default="dodphi",
        help="multipath Dataset for the model"
    )
    parser.add_argument(  # 添加数据集选择参数
        "--multipath_name4", type=str,
        choices=["mnist", "cifar", "custom", "RGB2CIR", "RGB2power", "RGB2dodtheta", "RGB2dodphi", "RGB2delay"],
        default="dodtheta",
        help="multipath Dataset for the model"
    )
    parser.add_argument(  # 添加设备选择参数
        "--device", type=str, default="cuda:1",
        choices=["cpu", "cuda"],
        help="Device to train the model on"
    )
    parser.add_argument(  # 添加随机种子参数
        "--seed", type=str, default=42,
        help="Seed for Reproducibility"
    )
    parser.add_argument(  # 添加设备选择参数
        "--batch_size", type=int, default=1,
    )
    parser.add_argument(  # 添加设备选择参数
        "--epochs_vqgan", type=int, default=0,
    )
    parser.add_argument(  # 添加设备选择参数
        "--epochs_transformer", type=int, default=0,
    )
    args = parser.parse_args()  # 解析命令行参数

    # 7. 配置文件加载
    with open(args.config_path) as f:  # 打开配置文件
        config = yaml.load(f, Loader=yaml.FullLoader)  # 用yaml加载配置

    main(args, config)  # 执行主函数