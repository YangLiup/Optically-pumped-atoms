import numpy as np
import matplotlib.pyplot as plt
t=1
# 使用 numpy 生成 X 和 Y 的数据
L = np.arange(20, 400, 0.1)  # 生成从 -5 到 5 的 100 个等间距的点
# R = np.arange(1, 2000, 0.1)  # 同样为 Y 轴生成相同范围的点
# L=100
R=20
St=t/(2*R)
f=1+L/(200*R)
N=-0.048/np.sqrt(0.5*L/R)+0.329/(0.5*L/R)-0.053/(0.5*L/R)**2
Sa=(1+4*N*St)/f


# 创建 3D 图形
fig = plt.figure()
ax  = fig.add_subplot(111)
ax.plot(L,Sa)
# 设置坐标轴标签
ax.set_xlabel('L')
ax.set_ylabel('Sa')

# 显示图形
plt.show()

