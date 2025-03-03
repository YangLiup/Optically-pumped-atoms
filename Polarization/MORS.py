import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

t = []
sx = []
sy = []
try:
    file = open('D:\python\pythonProject\Optically_pumped_atoms\Polarization\X.txt', 'r')
except FileNotFoundError:
    print('File is not found')
else:
    lines = file.readlines()
    for line in lines:
        a = line.split()
        x = a[0]
        y = a[1]
        t.append(float(x))
        sx.append(float(y))
file.close()

try:
    file = open('Y.txt', 'r')
except FileNotFoundError:
    print('File is not found')
else:
    lines = file.readlines()
    for line in lines:
        a = line.split()
        y = a[1]
        sy.append(float(y))
file.close()

t=np.array(t)
sx=np.array(sx)
sy=np.array(sy)

N=len(t)
T=t[-1]
s_fft = fft(sx+1j*sy)[:N//2]

freq = fftfreq(N, T/N)[:N//2]

xdata = freq
ydata= np.abs(s_fft)

def population(P,m):
    sum=0
    for mm in [-2,-1,0,1,2]:
        sum=sum+(1+P)**mm/(1-P)**mm
    return (1+P)**m/(1-P)**m/sum
def Z(F,m,P):
    return (F*(F+1)-(m+1)*m)*(population(P,m+1)-population(P,m))

# ,Gamma3,Gamma4
def func(freq,P,A,freq1,Delta,Gamma1,Gamma2,Gamma3,Gamma4):
    freq2=freq1-2*Delta
    freq3=freq1-4*Delta
    freq4=freq1-6*Delta
    return A*np.abs(Z(2,1,P)/(1j*(-freq+freq1)-Gamma1))
    # return A*np.abs(Z(2,1,P)/(1j*(-freq+freq1)-Gamma1)+Z(2,0,P)/(1j*(-freq+freq2)-Gamma2)+Z(2,-1,P)/(1j*(-freq+freq3)-Gamma3)+Z(2,-2,P)/(1j*(-freq+freq4)-Gamma4))


# 利用curve_fit作简单的拟合，popt为拟合得到的参数,pcov是参数的协方差矩阵
popt_1, pcov = curve_fit(func, xdata, ydata,bounds=(0, [0.999999, np.inf, np.inf,np.inf,np.inf,np.inf,np.inf,np.inf]))
print(popt_1) 
plt.figure()
plt.plot(xdata, func(xdata, *popt_1), 'r-', label='fit_1')
plt.plot(xdata, ydata, 'b-')

# #绘制原始信号
# plt.subplot(211)
# plt.plot(t,s)

# #绘制频谱图
# plt.subplot(212)
# plt.plot(freq, np.abs(s_fft))
plt.savefig('Polarization_fit.png', dpi=1000)
plt.show()
