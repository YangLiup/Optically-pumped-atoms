import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

z = np.array(pd.read_csv('data/dt=0.001/Interactive-1.interactive (Gammamdata).csv'))
PP = np.array(pd.read_csv('data/dt=0.001/Interactive-1.interactive (PPdata).csv'))
DD = np.array(pd.read_csv('data/dt=0.001/Interactive-1.interactive (DDdata).csv'))


for i in np.arange(0,1,1):
    deleter=[n for n in range(0, len(PP), 2)]
    PP=np.delete(PP, deleter)
    z=np.delete(z, deleter)
    DD=np.delete(DD, deleter)

eta=(5+3*PP**2)/(1-PP**2)
q1=2*(3+PP**2)/(1+PP**2)
fm1 = 2*q1/(q1-4)
fp1 = (q1-4)**2*(q1+4)/(2*16*q1**3) #*(q1+4)/(q1-4)

kappa1=z/fm1
# kappa2=fp1/DD*eta
# kappa2=(5+PP**2)/8

# plt.figure()
# plt.plot(PP,(kappa2-kappa1)/(kappa1+kappa2))
# plt.figure()
# plt.plot(PP,kappa1)
# plt.plot(PP,kappa2)
# plt.legend(['DM','草稿'])
# 定义目标函数
def func(x,a1,a2,a3):
    return (1/8)*(1+a1*x+a2*x**2+a3*x**3)

# 这部分生成样本点，对函数值加上高斯噪声作为样本点
# [0, 4]共50个点
xdata = PP
ydata=kappa1


# 利用curve_fit作简单的拟合，popt为拟合得到的参数,pcov是参数的协方差矩阵
popt_1, pcov = curve_fit(func, xdata, ydata)
print(popt_1) #[2.52560138 1.44826091 0.53725085]
plt.figure()
plt.plot(xdata, func(xdata, *popt_1), 'r-', label='fit_1')
plt.plot(xdata, kappa1, 'b-', label='data')

# 限定参数范围：0<=a<=3, 0<=b<=1, 0<=c<=0.5
# popt_2, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))
# print(popt_2) # [2.37032282 1.         0.39448271]
# plt.plot(xdata, func(xdata, *popt_2), 'g--', label='fit_2')
# plt.legend()