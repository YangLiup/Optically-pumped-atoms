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

z = np.array(pd.read_csv('D:\Software\python\pythonProject\Optically-pumped-atoms\Optically-pumped-atoms\data\dt=0.001\z_I.csv'))
PP = np.array(pd.read_csv('D:\Software\python\pythonProject\Optically-pumped-atoms\Optically-pumped-atoms\data\dt=0.001\PP_I.csv'))
DD = np.array(pd.read_csv('D:\Software\python\pythonProject\Optically-pumped-atoms\Optically-pumped-atoms\data\dt=0.001\DD_I.csv'))

for i in np.arange(0,10,1):
    deleter=[n for n in range(0, len(PP), 2)]
    PP=np.delete(PP, deleter)
    z=np.delete(z, deleter)
    DD=np.delete(DD, deleter)

dz=np.zeros(len(z)-101)
for i in np.arange(100,len(z)-1,1):
    dz[i]=(z[i+1]-z[i])/(PP[i+1]-PP[i])


plt.figure()
plt.plot(PP[0:len(PP)-1],dz)
plt.title("I=31/2")
plt.show()


# 限定参数范围：0<=a<=3, 0<=b<=1, 0<=c<=0.5
# popt_2, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))
# print(popt_2) # [2.37032282 1.         0.39448271]
# plt.plot(xdata, func(xdata, *popt_2), 'g--', label='fit_2')
# plt.legend()
