from typing import Any
import numpy as np
import matplotlib.pyplot as plt


class ACC:
    def __init__(self, label) -> None:
        self.__label = label

    def __call__(self, pred) -> Any:
        return self.calculate_clustering_accuracy(self.__label, pred)

    def calculate_clustering_accuracy(self, y_true, y_pred):
        assert y_pred.size == y_true.size
        # 统计每个簇中真实标签出现的次数
        unique_clusters = np.unique(y_pred)
        accuracy = 0.0

        for cluster in unique_clusters:
            cluster_indices = np.where(y_pred == cluster)[0]
            true_labels_in_cluster = y_true[cluster_indices]

            # 统计真实标签的出现次数
            unique_labels, counts = np.unique(
                true_labels_in_cluster, return_counts=True
            )
            # 找到出现次数最多的真实标签作为该簇的预测标签
            majority_label = unique_labels[np.argmax(counts)]

            # 计算匹配正确的样本数量
            accuracy += np.max(counts)

        # 计算聚类精度
        clustering_accuracy = accuracy / len(y_true)
        return clustering_accuracy


plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

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
line_styles = ["-"] * 15


def muti_lines(
    lines: list, x_Datas, Datas, x_label, y_label, labels, title, path_back=""
):
    plt.figure(figsize=(20, 10), dpi=100)
    for t in lines:
        plt.plot(
            x_Datas[t], Datas[t], c=colors[t], linestyle=line_styles[t], label=labels[t]
        )
        plt.scatter(x_Datas[t], Datas[t], c=colors[t])
    # y_ticks = range(50)
    # plt.yticks(y_ticks[::5])

    plt.title(title, fontdict={"size": 20})
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlabel(x_label, fontdict={"size": 16})
    plt.ylabel(y_label, fontdict={"size": 16})
    if path_back != "":
        plt.savefig(path_back)
    else:
        plt.show()


def plot_accuracy(accuracy,test_acc):
    iterations=range(1, len(accuracy) + 1)
    plt.plot(iterations, accuracy, marker="o")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy for test: {test_acc*100:.3f}%")
    plt.grid(True)
    plt.show()
