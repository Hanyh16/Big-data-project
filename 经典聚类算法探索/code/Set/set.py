import numpy as np
import argparse

import torch

# from typing import List, Optional
# import os

from os import environ

environ["QT_DEVICE_PIXEL_RATIO"] = "0"
environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
environ["QT_SCREEN_SCALE_FACTORS"] = "1"
environ["QT_SCALE_FACTOR"] = "1"


# 设置将模型放在GPU上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 设置随机种子
def setup_seed(seed):
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True


setup_seed(114514)

num_classes = 10

parser = argparse.ArgumentParser(description="cluster config")
data_parser = argparse.ArgumentParser(description="general data config")
k_parser = argparse.ArgumentParser(description="kmeans cluster config")
g_parser = argparse.ArgumentParser(description="gmm cluster config")

# 数据地址
data_parser.add_argument(
    "--dir_path", default=r"./data/mnist_", type=str, help="the path of datas' dir"
)

data_parser.add_argument(
    "--file_format", default=r".csv", type=str, help="the format for the data format"
)

# 输出地址
data_parser.add_argument(
    "--outfile", default="temp_0.1.csv", type=str, help="Output file name"
)

data_parser.add_argument(
    "--matr", default="results/acc_matr.npz", help="Accuracy matrix file name"
)

# 数据信息
data_parser.add_argument(
    "--n_classes", default=num_classes, help="total classes for cifar10", type=int
)

data_parser.add_argument(
    "--nrows", default=None, help="the num of lines read from the csv"
)

data_parser.add_argument(
    "--shuffle",
    default=True,
    type=bool,
    help="whether reorder the train dataset before use it",
)

data_parser.add_argument(
    "--resize", default=True, type=bool, help="whether resize the data(img) to 1*28*28"
)

data_parser.add_argument(
    "--pca", default=70, type=int, help="the dimention after using the pca method(if using pca, >0)"
)

data_parser.add_argument(
    "--img_size", default=(1, 28, 28), type=tuple, help="the img size"
)

data_parser.add_argument(
    "--device",
    default=device,
    type=str,
    help="the model's position when processing the row data",
)

parser.add_argument(
    "--data", default=data_parser.parse_args(), help="the config for data IO"
)

# k_means 训练设置
k_parser.add_argument("--max_iters", default=30, type=int, help="Number of epochs")

k_parser.add_argument(
    "--n_clusters", default=num_classes, help="total classes for cifar10", type=int
)

k_parser.add_argument(
    "--k_init",
    default="kmeans++",
    help="the function used to initialize the centroids",
    type=str,
)

parser.add_argument(
    "--k_args", default=k_parser.parse_args(), help="the config for k-means"
)

# k_parser.add_argument("--batch_size", default=64, type=int, help="Mini batch size")

# gmm 参数设置

g_parser.add_argument("--max_iters", default=30, type=int, help="Number of epochs")

g_parser.add_argument(
    "--n_clusters", default=num_classes, help="total classes for cifar10", type=int
)

g_parser.add_argument(
    "--tol",
    default=1e-4,
    help=" tolerance or threshold to determine when to stop iterations",
    type=float,
)

g_parser.add_argument(
    "--init_cen",
    default="random",
    help="the function used to initialize the centroids",
    type=str,
)

g_parser.add_argument(
    "--covs_type",
    default="general",
    help="the function used to initialize the covs",
    type=str,
)

parser.add_argument(
    "--g_args", default=g_parser.parse_args(), help="the config for gmm"
)

if __name__ == "__main__":
    pass
