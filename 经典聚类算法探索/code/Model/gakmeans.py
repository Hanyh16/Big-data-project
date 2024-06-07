import sys
#添加上级目录
sys.path.append(r"code\Set")

import numpy as np
from tqdm import trange
from scipy.spatial.distance import cdist


class GAKMeans:
    # 构造方法
    def __init__(self, k, pop_size=50, gen_size=20, mut_rate=0.1, cross_rate=0.8):
        # 初始化类的属性
        self.k = k  # 聚类个数
        self.pop_size = pop_size  # 种群大小
        self.gen_size = gen_size  # 迭代次数
        self.mut_rate = mut_rate  # 变异概率
        self.cross_rate = cross_rate  # 交叉概率
        self.centroids = None  # 聚类中心
        self.labels = None  # 聚类标签
        self.fitness = None  # 聚类误差
        self.population = None

    # 私有方法，计算一个个体的适应度
    def __fitness(self, individual, X):
        # 将个体的基因解码为聚类中心
        centroids = individual.reshape(self.k, -1)
        # 计算数据点到最近的聚类中心的距离
        dist = np.min(cdist(X, centroids, "euclidean"), axis=1)
        # 返回聚类误差的平方和
        return np.sum(dist**2)

    # 私有方法，对一个个体进行变异操作
    def __mutate(self, individual):
        # 遍历每个基因
        for i, j in enumerate(individual):
            # 以一定的概率进行变异
            if np.random.rand() < self.mut_rate:
                # 在原基因的基础上加上一个随机数
                individual[i] += np.random.normal()
        # 返回变异后的个体
        return individual

    # 私有方法，对两个个体进行交叉操作
    def __crossover(self, individual1, individual2):
        # 遍历每个基因
        for i, j in enumerate(individual1):
            # 以一定的概率进行交叉
            if np.random.rand() < self.cross_rate:
                # 交换两个个体的基因
                individual1[i], individual2[i] = individual2[i], individual1[i]
        # 返回交叉后的两个个体
        return individual1, individual2

    # 私有方法，从一个种群中选择最优的个体
    def __select(self, X):
        # 计算种群中每个个体的适应度
        fitnesses = np.array(
            [self.__fitness(individual, X) for individual in self.population]
        )
        # 按照适应度从小到大排序
        indices = np.argsort(fitnesses)
        best = self.population[indices[0]]
        # 减小种群
        self.population = self.population[indices[: self.pop_size]]
        # 选择适应度最小的个体
        return best

    # 公有方法，对数据集进行聚类，更新类的属性
    def fit(self, X, acc_fun):
        # 初始化种群
        self.population = np.random.randn(self.pop_size, self.k * X.shape[1])
        acc = []
        # 进行迭代
        with trange(self.gen_size) as t:
            for _ in t:
                # 对种群中的每个个体进行变异
                mul = np.array(
                    [self.__mutate(individual) for individual in self.population]
                )
                # 对种群中的每对个体进行交叉
                children = []
                for i in range(0, self.pop_size, 2):
                    ch1, ch2 = self.__crossover(
                        self.population[i], self.population[i + 1]
                    )
                    children.append(ch1)
                    children.append(ch2)
                # 合并种群
                self.population = np.concatenate(
                    (self.population, mul, children), axis=0
                )
                # 选择最优的个体
                best = self.__select(X)
                self.centroids = best.reshape(self.k, -1)  # 聚类中心
                acc.append(acc_fun(self.predict(X)))
                # 打印最优个体的适应度
                t.set_postfix(acc=f"{acc[-1]*100:.3f}%")  # 预测准确率
                # print(f"Best fitness: {self.__fitness(best,X)}")
        # 更新类的属性
        self.labels = np.argmin(cdist(X, self.centroids, "euclidean"), axis=1)  # 聚类标签
        self.fitness = self.__fitness(best, X)  # 聚类误差
        return acc

    # 公有方法，对数据点进行聚类，返回聚类标签
    def predict(self, X):
        # 计算每个样本到聚类中心的距离
        distances = cdist(X, self.centroids, metric="euclidean")
        # 分配样本到最近的聚类中心
        labels = np.argmin(distances, axis=1)
        return labels
