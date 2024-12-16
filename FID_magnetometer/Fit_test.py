import numpy as np
from scipy.optimize import curve_fit
 
# 定义模型函数
def func(t, omega, phi, A,Gamma):
    return A* np.sin(omega*t+phi)*np.exp(-Gamma*t)
 
# 生成模拟数据
xdata = np.arange(0, 10,0.01)
y = func(xdata ,15, -2 ,1,0.005)
np.random.seed(0) # 设置随机种子以确保结果可复现性。
y_noise = y + 0.05 * np.random.normal(size=xdata.size)
 
# 使用curve_fit进行拟合
popt, pcov = curve_fit(func,xdata,y_noise)
plt.plot(xdata, func(xdata, *popt), 'r-', label='fit_1')
plt.plot(xdata, y_noise  , 'b-', label='data',linewidth='0.5')
print("Optimal parameters are: a=%f,b=%f,c=%f" % (popt[0], popt[1], popt[2]))