from Tool.utils import *
from Set.set import *
from Model.gakmeans import GAKMeans
from Data.dataloader import Dataloader
import time

args = parser.parse_args()

args.k_args.k_init = "kmeans++"
# args.k_args.k_init = "random"
# args.k_args.max_iter = 20
# args.data.nrows = 500
# args.data.pca = -1

dl = Dataloader(args.data)

# 初始化ACC方法，用于计算聚类精度，标签变量已做私有化
ACC_train = ACC(dl.y_train)
ACC_test = ACC(dl.y_test)

# 初始化并训练K-Means模型
kmeans = GAKMeans(10)
train_time = time.time()
acc = kmeans.fit(dl.X_train, ACC_train)
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

plot_accuracy(acc, test_acc=test_accuracy)
