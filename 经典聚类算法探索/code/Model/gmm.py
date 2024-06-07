import sys
#添加上级目录
sys.path.append(r"code\Set")
sys.path.append(r"code\Model")

import numpy as np
from scipy.stats import multivariate_normal
from tqdm import trange
from set import g_parser
from KMeans import KMeans


class GMM:
    def __init__(self, config=None):
        if config == None:
            config = g_parser.parse_args()
        self.g_args = config
        self.n_clusters = config.n_clusters
        self.max_iters = config.max_iters
        self.tol = config.tol
        self.weights = np.ones(self.n_clusters) / self.n_clusters
        self.means = None
        self.covs = None

    def fit(self, X, acc_fun):
        self.init_centroids(X, self.g_args.init_cen, acc_fun)
        self.init_covs(X, self.g_args.covs_type)

        acc = []
        with trange(self.max_iters) as t:
            for _ in t:
                # E-step
                posteriors = self._expectation(X)

                # M-step
                self._maximization(X, posteriors)

                acc.append(acc_fun(self.predict(X)))
                t.set_postfix(acc=f"{acc[-1]*100:.3f}%")  # 预测准确率

        return acc

    def _expectation(self, X):
        posteriors = []
        for i in range(self.n_clusters):
            posterior = self.weights[i] * multivariate_normal.pdf(
                X, self.means[i], self.covs[i], allow_singular=True
            )

            posteriors.append(posterior)
        posteriors = np.array(posteriors).T + 1e-200
        posteriors = posteriors / np.sum(posteriors, axis=1, keepdims=True)
        return posteriors

    def _maximization(self, X, posteriors):
        for i in range(self.n_clusters):
            total_weight = np.sum(posteriors[:, i])
            self.weights[i] = total_weight / len(X)
            self.means[i] = np.dot(posteriors[:, i], X) / total_weight

            if self.g_args.covs_type == "general":
                self.covs[i] = (
                    np.dot(posteriors[:, i] * (X - self.means[i]).T, X - self.means[i])
                    / total_weight
                )

            elif self.g_args.covs_type == "diagonal_unequal":
                self.covs[i] = np.diag(
                    np.sum(posteriors[:, i] * (X - self.means[i]).T ** 2, axis=1)
                    / total_weight
                )

            elif self.g_args.covs_type == "diagonal_equal":
                # t = np.random.randint(0, posteriors.shape[1])
                t = 0
                diag_t = (
                    np.sum(posteriors[:, i] * (X - self.means[i])[:, t] ** 2)
                    / total_weight
                )
                self.covs[i] = np.eye(X.shape[1]) * diag_t

    def predict(self, X):
        posteriors = self._expectation(X)
        return np.argmax(posteriors, axis=1)

    def init_centroids(self, X, cen_type="kmeans++", acc_fun=None):
        if cen_type == "random":
            # 从数据集 X 中随机选择若干个样本作为聚类中心
            self.means = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        elif cen_type == "kmeans++":
            i_range = X.shape[0]

            self.means = [X[np.random.randint(i_range)]]
            # 从数据集 X 中随机选择一个样本作为第一个聚类中心

            for _ in range(1, self.n_clusters):
                # 循环选取其余的聚类中心

                # 更新样本点到最近聚类中心的最短距离
                distances = np.array(
                    [
                        min([np.linalg.norm(x - c) for c in self.means])
                        for x in X[:i_range]
                    ]
                )
                # 计算每个样本点到最近聚类中心的距离的平方，并更新最小距离列表

                # 计算每个样本点被选择作为下一个聚类中心的概率
                probs = distances / distances.sum()

                i = np.argmax(probs)
                # 将选择的样本点作为新的聚类中心
                self.means.append(X[i])

            # 将结果转换为 NumPy 数组
            self.means = np.array(self.means)
        elif cen_type == "KMeans":
            kmeans = KMeans()
            kmeans.k_args.k_init = "random"
            kmeans.k_args.max_iters = 20
            kmeans.fit(X, acc_fun)
            self.means = kmeans.centroids

    def init_covs(self, X, cov_type="general"):
        n_samples, n_features = X.shape
        if cov_type == "diagonal_equal":
            self.covs = np.array(
                [np.eye(n_features) * np.var(X, axis=0)] * self.n_clusters
            )
        elif cov_type == "diagonal_unequal":
            self.covs = np.array(
                [np.diag(np.var(X, axis=0) + np.random.rand(n_features) * 0.1)]
                * self.n_clusters
            )
        elif cov_type == "general":
            self.covs = np.array([np.cov(X.T)] * self.n_clusters)

