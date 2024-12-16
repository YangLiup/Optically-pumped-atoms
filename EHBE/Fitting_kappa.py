import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

# z = np.array(pd.read_csv('data/dt=0.001/z.csv'))
# PP = np.array(pd.read_csv('data/dt=0.001/PP1.csv'))
# DD = np.array(pd.read_csv('data/dt=0.001/DD1.csv'))


# z = np.array(pd.read_csv('data/dt=0.001/zz.csv'))
# PP = np.array(pd.read_csv('data/dt=0.001/PP2.csv'))
# DD = np.array(pd.read_csv('data/dt=0.001/DD2.csv'))

z = np.array(pd.read_csv('D:\python\pythonProject\Optically_pumped_atoms\data\dt=0.001\z.csv'))
PP = np.array(pd.read_csv('D:\python\pythonProject\Optically_pumped_atoms\data\dt=0.001\PP1.csv'))
DD = np.array(pd.read_csv('D:\python\pythonProject\Optically_pumped_atoms\data\dt=0.001\DD1.csv'))


for i in np.arange(0,10,1):
    deleter=[n for n in range(0, len(PP), 2)]
    PP=np.delete(PP, deleter)
    z=np.delete(z, deleter)
    DD=np.delete(DD, deleter)

eta=(5+3*PP**2)/(1-PP**2)
q1=2*(3+PP**2)/(1+PP**2)
fm = 2*q1/(q1-4)
fp = (q1-4)**2*(q1+4)/(2*16*q1**3) #*(q1+4)/(q1-4)

# q2 = 2 * (19 + 26 * PP ** 2 + 3 * PP ** 4) / (3 + 10 * PP ** 2 + 3 * PP ** 4)
# eta=(q2+6)/(q2-6)
# fp =  (q2-6)**2*(q2+6)/(2*36*q2**3)#*(q2+6)/(q2-6)
# fm=2*q2/(q2-6)#*(q2-6)/(q2+6)

# q3 = 2 * (11 + 35 * PP ** 2 + 17 * PP ** 4 + PP ** 6) / (1 + 7 * PP ** 2 + 7 * PP ** 4 + PP ** 6)
# eta=(q3+8)/(q3-8)
# fp = (q3-8)**2*(q3+8)/(2*64*q3**3)#*(q3+8)/(q3-8)
# fm =  2*q3/(q3-8)#*(q3-8)/(q3+8)

kappa1=z/fm*eta
kappa2=fp/DD*eta
kappa=(kappa1+kappa2)/2
# kappa2=(5+PP**2)/8

# plt.figure()
# plt.plot(PP,(kappa2-kappa1)/(kappa1+kappa2))
# plt.figure()
# plt.plot(PP,kappa1)
# plt.plot(PP,kappa2)
# plt.legend(['DM','草稿'])
# 定义目标函数
#I=3/2, 5/8
#I=5/2, 14/27
#I=7/2, 15/32
# def func(x,a2,a4,a6,a8,a10,a12):
#     return (5/8)*(1+a2*x**2+a4*x**4+a6*x**6+a8*x**8+a10*x**10+a12*x**12)
def func(x,a2,a4,a6,a8,a10):
    return (5/8)*(1+a2*x**2+a4*x**4+a6*x**6+a8*x**8+a10*x**10)


xdata = PP
ydata=kappa


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
mean = np.mean(ydata)  # 1.y mean
ss_tot = np.sum((ydata - mean) ** 2)  # 2.total sum of squares
ss_res = np.sum((ydata - func(xdata, *popt_1)) ** 2)  # 3.residual sum of squares
r_squared = 1 - (ss_res / ss_tot)  # 4.r squared
r_squared 