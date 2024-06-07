import sys
#添加上级目录
sys.path.append(r"code\Set")

import pandas as pd
import numpy as np
# from .. Set import set
from set import data_parser
import matplotlib.pyplot as plt
import torch
from torchvision import models, transforms
from sklearn.decomposition import PCA


class Dataloader:
    def __init__(self, config=None) -> None:
        if config == None:
            config = data_parser.parse_args()
        if config.file_format == ".csv":
            # 读取训练集和测试集数据
            train_data = pd.read_csv(
                config.dir_path + "train" + config.file_format,
                header=0,
                dtype=np.float32,
                nrows=config.nrows,
            )
            test_data = pd.read_csv(
                config.dir_path + "test" + config.file_format,
                header=0,
                dtype=np.float32,
                nrows=config.nrows,
            )
        self.config = config
        # column_names = train_data.columns.tolist()
        # print(column_names)
        # 分割特征和标签
        self.X_train = train_data.iloc[:, 1:].values  # 训练集特征
        self.y_train = train_data.iloc[:, 0].values  # 训练集标签

        self.X_test = test_data.iloc[:, 1:].values  # 测试集特征
        self.y_test = test_data.iloc[:, 0].values  # 测试集标签

        if config.pca > 0:
            self.PCA(config.pca)

        # self.X_train = self.pre_train(self.X_train)
        # self.X_test = self.pre_train(self.X_test)

    def get_img(self, index, kind="train"):
        if kind == "test":
            pixel_values = self.X_test[index]
        else:
            pixel_values = self.X_train[index]

        # 将一维像素数据转换为二维数组，假设图像大小为 28x28 像素
        image_array = np.reshape(pixel_values, (28, 28)).astype(np.uint8)
        return image_array

    def pre_train(self, row_data):
        # 将数据转换为 PyTorch 张量
        tensor_data = torch.tensor(row_data, dtype=torch.float32)

        # 创建预处理转换
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
            ]
        )

        # 构建预训练的 ResNet 模型
        model = models.resnet18(pretrained=True)
        # Modify the first layer weights to accept one channel input
        model.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        model.to(self.config.device)
        model.eval()  # 将模型设置为评估模式

        # 对数据进行预处理和提取特征
        # 将数据还原成图像大小
        tensor_data = tensor_data.view(-1, 28, 28)
        tensor_data = tensor_data.unsqueeze(1)  # 添加一个通道维度

        # 预处理图像数据
        tensor_data = preprocess(tensor_data).to(self.config.device)

        # 使用模型提取特征
        with torch.no_grad():
            tensor_data = model(tensor_data)
            tensor_data = tensor_data.squeeze().cpu().numpy()  # 将特征转换为 NumPy 数组

        return tensor_data

    def PCA(self, n_components=100):
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=n_components)  # Specify the number of components for PCA
        pca.fit(self.X_train)
        self.X_train = pca.transform(self.X_train)
        self.X_test = pca.transform(self.X_test)


if __name__ == "__main__":
    config = data_parser.parse_args()
    config.pca = 2
    config.nrows = 10
    dl = Dataloader(config)
    # for i in range(10000):
    #     # plt.imshow(np.transpose(X_train[i], (1, 2, 0)))
    #     plt.imshow(dl.get_img(i), cmap="gray")  # 使用灰度颜色映射
    #     plt.axis("off")  # 关闭坐标轴
    #     plt.title(f"label:{dl.y_train[i]}")
    #     plt.show()
