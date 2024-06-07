from Tool.utils import *
from Set.set import *
from Model.KMeans import KMeans
from Model.gmm import GMM
from Data.dataloader import Dataloader
import time

args = parser.parse_args()


# args.k_args.k_init = "kmeans++"
# args.k_args.k_init = "random"
# args.g_args.max_iters = 50
# args.data.nrows = 1500
args.data.pca = 70
args.g_args.max_iters = 30
args.g_args.covs_type = "diagonal_equal"
# args.g_args.covs_type = "diagonal_unequal"
args.g_args.covs_type = "general"
args.g_args.init_cen = "kmeans++"

args.k_args.k_init = "random"
# args.k_args.k_init = "kmeans++"
args.k_args.max_iters = 20

dl = Dataloader(args.data)

# 初始化ACC方法，用于计算聚类精度，标签变量已做私有化
ACC_train = ACC(dl.y_train)
ACC_test = ACC(dl.y_test)

# 初始化K-Means并训练

kmeans = KMeans(args.k_args)
train_time = time.time()
acck = kmeans.fit(dl.X_train, ACC_train)
train_time = time.time() - train_time

# 在训练集和测试集上进行预测
train_pred = kmeans.predict(dl.X_train)
test_pred = kmeans.predict(dl.X_test)

# 使用ACC函数做预测结果的重新标记，并计算准确率
train_accuracy = ACC_train(train_pred)
test_accuracy = ACC_test(test_pred)

print(f"训练集上的聚类精度：{train_accuracy * 100:.3f}%")
print(f"测试集上的聚类精度：{test_accuracy * 100:.3f}%")
print(f"train time: {train_time:.3f}s")

plot_accuracy(acck, test_acc=test_accuracy)

# 初始化并训练K-Means/gmm模型
gmm = GMM(args.g_args)
train_time = time.time()
accg = gmm.fit(dl.X_train, ACC_train)
train_time = time.time() - train_time

# 在训练集和测试集上进行预测
train_pred = gmm.predict(dl.X_train)
test_pred = gmm.predict(dl.X_test)

# 使用ACC函数做预测结果的重新标记，并计算准确率
train_accuracy = ACC_train(train_pred)
test_accuracy = ACC_test(test_pred)

print(f"训练集上的聚类精度：{train_accuracy * 100:.3f}%")
print(f"测试集上的聚类精度：{test_accuracy * 100:.3f}%")
print(f"train time: {train_time:.3f}s")

plot_accuracy(accg, test_acc=test_accuracy)

# kmeans = KMeans(args.k_args)

# kmeans.centroids = gmm.means
# test_accuracy_k = ACC_test(kmeans.predict(dl.X_test))

# plot_accuracy(acc, test_acc=test_accuracy_k)
