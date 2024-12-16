# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2024年03月07日
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#计算失谐相关的耦合因子
def voigt_profile(x, sigma, gamma):
    """
    Calculate the Voigt profile.

    Parameters:
        x (array-like): The x-values at which to calculate the profile.
        sigma (float): The Gaussian standard deviation.
        gamma (float): The Lorentzian full-width at half-maximum.

    Returns:
        array-like: The Voigt profile values at the specified x-values.
    """
    z = (x + 1j * gamma) / (sigma * np.sqrt(2))
    v = wofz(z).imag / (sigma * np.sqrt(2 * np.pi))
    return v


Gammap=2        #压强展宽GHz
Gammad=0.5      #多普勒展宽GHz
sigma=Gammad/(2*np.sqrt(2*np.log(2)))

#delta_nv为失谐
def chia(delta_nv):
    chia=-voigt_profile(delta_nv+2.3,sigma,Gammap)/4-3/4*voigt_profile(delta_nv+3.1,sigma,Gammap)
    return chia
def chib(delta_nv):
    chib=5*voigt_profile(delta_nv+2.3-6.8,sigma,Gammap)/4-1/4*voigt_profile(delta_nv+3.1-6.8,sigma,Gammap)
    return chib

#定义旋光角
# def phi(delta_nv,ax,bx):
#     phi=chia(delta_nv)*ax+chib(delta_nv)*bx
#     return phi

#theta为极化P倒下的角度，假设纵向是一个自旋温度分布且拉莫频率远大于自旋交换碰撞的频率
def phi(delta_nv,P,theta):  
    q=2*(3+P**2)/(1+P**2)
    eta=(q+4)/(q-4)
    ax=q*P/2*eta/(eta+1)*np.sin(theta)
    bx=-q*P/2/(eta+1)*np.sin(theta)
    phi=10*chia(delta_nv)*ax+10*chib(delta_nv)*bx
    return phi

#绘制扫频曲线
delta_nv=np.arange(-30,30,0.01)
fig = plt.figure()
plt.rc('font',family='Times New Roman')
ax1= fig.add_subplot(111)
ax1.plot(delta_nv,phi(delta_nv,0.5,1.5))
plt.show()
