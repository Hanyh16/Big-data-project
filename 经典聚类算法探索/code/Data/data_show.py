import sys

# 添加上级目录
sys.path.append(r"code\Set")

from dataloader import Dataloader
from set import *
from matplotlib import pyplot as plt
import numpy as np

np.random.seed(114514)

config = data_parser.parse_args()
config.pca = 2

dl = Dataloader(config)

# cmap = plt.cm.get_cmap("tab10", 10)

colors = [
    "#3e82fc",
    "#f54748",
    "#ff8f00",
    "#1abc9c",
    "#e67e22",
    "#8e44ad",
    "#2ecc71",
    "#c0392b",
    "#f1c40f",
    "#3498db",
    "#e74c3c",
    "#2c3e50",
    "#9b59b6",
    "#16a085",
    "#f39c12",
]

num = 6000

random_index = np.random.randint(0, dl.X_train.shape[0], num)

plt.scatter(
    dl.X_train[random_index, 0],
    dl.X_train[random_index, 1],
    c=[colors[int(dl.y_train[random_index[i]])] for i in range(num)],
)
plt.title(f"{num} images' distribution(with pca to 2 dim)")

plt.show()
