import numpy as np
import matplotlib.pyplot as plt

# 生成数列
num_values = 30 + 1
values = np.zeros(num_values)
values[0] = 0.1

for i in range(1, num_values):
    values[i] = values[i - 1] * 0.99

# 绘制图像
plt.plot(values, marker="o", linestyle="-")
plt.title("Exponential Decay")
plt.xlabel("Index")
plt.ylabel("Value")
plt.grid(True)
plt.show()
