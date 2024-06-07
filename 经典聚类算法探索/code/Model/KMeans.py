import sys
#添加上级目录
sys.path.append(r"code\Set")

import numpy as np
from scipy.spatial.distance import cdist
from tqdm import trange
from set import k_parser


# 实现K-Means算法
class KMeans:
    def __init__(self, config=None):
        if config == None:
            config = k_parser.parse_args()
        self.k_args = config
        self.n_clusters = config.n_clusters
        self.max_iters = config.max_iters
        self.centroids = []
        self.labels = []

    def fit(self, X, acc_fun):
        # 初始化聚类中心
        self.init_centroids(X, opt=self.k_args.k_init)
        acc = []
        with trange(self.max_iters) as t:
            for _ in t:
                # 预测x类别
                labels = self.predict(X)
                acc.append(acc_fun(labels))
                t.set_postfix(acc=f"{acc[-1]*100:.3f}%")  # 预测准确率
                # 更新聚类中心
                for i in range(self.n_clusters):
                    if np.sum(labels == i) > 0:
                        self.centroids[i] = np.mean(X[labels == i], axis=0)
        self.labels = labels
        return acc

    def predict(self, X):
        # 计算每个样本到聚类中心的距离
        distances = cdist(X, self.centroids, metric="euclidean")
        # 分配样本到最近的聚类中心
        labels = np.argmin(distances, axis=1)
        return labels

    def init_centroids(self, X, opt="random"):
        if opt == "random":
            # 从数据集 X 中随机选择若干个样本作为聚类中心
            self.centroids = X[
                np.random.choice(X.shape[0], self.n_clusters, replace=False)
            ]
        elif opt == "kmeans++":
            i_range = X.shape[0]//4

            self.centroids = [X[np.random.randint(i_range)]]
            # 从数据集 X 中随机选择一个样本作为第一个聚类中心

            for _ in range(1, self.n_clusters):
                # 循环选取其余的聚类中心

                # 更新样本点到最近聚类中心的最短距离
                distances = np.array(
                    [
                        min([np.linalg.norm(x - c) for c in self.centroids])
                        for x in X[:i_range]
                    ]
                )
                # 计算每个样本点到最近聚类中心的距离的平方，并更新最小距离列表

                # 计算每个样本点被选择作为下一个聚类中心的概率
                probs = distances / distances.sum()
                # cumulative_probs = np.cumsum(probs)

                # # 生成一个随机数 r
                # r = np.random.rand()
                # # 根据 r 的值选择下一个聚类中心
                # i = np.searchsorted(cumulative_probs, r)
                i = np.argmax(probs)
                # 将选择的样本点作为新的聚类中心
                self.centroids.append(X[i])

            # 将结果转换为 NumPy 数组
            self.centroids = np.array(self.centroids)
