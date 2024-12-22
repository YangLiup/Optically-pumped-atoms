import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft


t=np.arange(0,10,0.1)
y1=0.5*np.exp(-0.5*t)+0.1*np.exp(-t)+0.1*np.exp(-1.5*t)
y2=0.7*np.exp(-0.6*t)
fft_y1=fft(y1)                          #快速傅里叶变换
fft_y2=fft(y2)                          #快速傅里叶变换



N=100
x = np.arange(N)             # 频率个数
half_x = x[range(int(N/2))]  #取一半区间
 
abs_y1=np.abs(fft_y1)                # 取复数的绝对值，即复数的模(双边频谱)
abs_y2=np.abs(fft_y2)                # 取复数的绝对值，即复数的模(双边频谱)


normalization_y1=abs_y1/N            #归一化处理（双边频谱）
normalization_y2=abs_y2/N            #归一化处理（双边频谱）                              

normalization_half_y1 = normalization_y1[range(int(N/2))]      #由于对称性，只取一半区间（单边频谱）
normalization_half_y2 = normalization_y2[range(int(N/2))]      #由于对称性，只取一半区间（单边频谱）


plt.figure()
plt.plot(t,y1)
plt.plot(t,y2,linestyle='dashed')
plt.show()

plt.figure()
plt.plot(half_x,normalization_half_y1)
plt.plot(half_x,normalization_half_y2,linestyle='dashed')
plt.show()


