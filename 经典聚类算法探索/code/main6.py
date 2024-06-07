from Tool.utils import *
from Set.set import *
from Model.others import hierarchical_clustering, spectral_clustering
from Data.dataloader import Dataloader
import time

args = parser.parse_args()

dl = Dataloader(args.data)

# 初始化ACC方法，用于计算聚类精度，标签变量已做私有化
ACC_train = ACC(dl.y_train)
ACC_test = ACC(dl.y_test)

# # 在训练集和测试集上进行预测
# train_pred = hierarchical_clustering(dl.X_train)
# test_pred = hierarchical_clustering(dl.X_test)

# # 使用ACC函数做预测结果的重新标记，并计算准确率
# train_accuracy = ACC_train(train_pred)
# test_accuracy = ACC_test(test_pred)

# print(f"层次聚类 训练集上的聚类精度：{train_accuracy * 100:.3f}%")
# print(f"层次聚类 测试集上的聚类精度：{test_accuracy * 100:.3f}%")
# print(f"train time: {train_time:.3f}s")

# 在训练集和测试集上进行预测
train_pred = spectral_clustering(dl.X_train)
test_pred = spectral_clustering(dl.X_test)

# 使用ACC函数做预测结果的重新标记，并计算准确率
train_accuracy = ACC_train(train_pred)
test_accuracy = ACC_test(test_pred)

print(f"谱聚类 训练集上的聚类精度：{train_accuracy * 100:.3f}%")
print(f"谱聚类 测试集上的聚类精度：{test_accuracy * 100:.3f}%")