import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt

# 定义模型函数
def func(t, omega, phi, A,Gamma):
    return A* np.sin(omega*t+phi)*np.exp(-Gamma*t)

# 生成模拟数据

dt=5.3333e-7
FID = np.array(pd.read_csv('data.csv')).flatten()
t = np.arange(0,len(FID)*dt,dt)

plt.plot(t,FID)
plt.xlim([0,0.001])
plt.show()
# # 使用curve_fit进行拟合
popt, pcov = curve_fit(func,t,FID)
 
# print("Optimal parameters are: a=%f,b=%f,c=%f" % (popt[0], popt[1], popt[2]))