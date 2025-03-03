import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
sx = np.array(pd.read_csv('D:\Optically-pumped-atoms\Polarization\data.csv'))


t=np.arange(0,len(sx),1)*0.002
sx=np.array(sx)

N=len(t)
T=t[-1]
s_fft = fft(sx)[:N//2]

freq = fftfreq(N, T/N)[:N//2]

xdata = freq
ydata = np.abs(s_fft)



plt.figure()
plt.plot(xdata, ydata, 'b-')

# #绘制原始信号
# plt.subplot(211)
# plt.plot(t,s)

# #绘制频谱图
# plt.subplot(212)
# plt.plot(freq, np.abs(s_fft))
plt.savefig('Polarization_fit.png', dpi=1000)
plt.show()
