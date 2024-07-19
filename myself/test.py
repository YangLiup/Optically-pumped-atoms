# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2024年03月07日
"""
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from scipy.special import wofz

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


Gammap=0.03        
Gammad=0.5       
sigma=Gammad/(2*np.sqrt(2*np.log(2)))

def chia(delta_nva2):
    chia=-voigt_profile(delta_nva2+2.3,sigma,Gammap)/4-3/4*voigt_profile(delta_nva2+3.1,sigma,Gammap)
    return chia
def chib(delta_nva2):
    chib=5*voigt_profile(delta_nva2+2.3-6.8,sigma,Gammap)/4-1/4*voigt_profile(delta_nva2+3.1-6.8,sigma,Gammap)
    return chib

def chip(delta_nva2,P):
    q=2*(3+P**2)/(1+P**2)
    eta=(3*P**2+5)/(1-P**2)
    return (eta*chia(delta_nva2)+chib(delta_nva2))/(eta+1)

def chim(delta_nva2,P):
    q=2*(3+P**2)/(1+P**2)
    eta=(3*P**2+5)/(1-P**2)
    return (chia(delta_nva2)-chib(delta_nva2))/(eta+1)

plt.style.use(['science','nature'])
with plt.style.context(['science','nature']):
    plt.rc('font',family='Times New Roman')
    delta_nva2=np.arange(-50,50,0.01)
    fig = plt.figure()
    plt.rc('font',family='Times New Roman')
    ax1= fig.add_subplot(111)

    p1,=ax1.plot(delta_nva2,chia(delta_nva2)/np.max(chia(delta_nva2)))
    p2,=ax1.plot(delta_nva2,chib(delta_nva2)/np.max(chia(delta_nva2)))
    ax1.plot(delta_nva2,np.zeros(len(delta_nva2)),linewidth=0.2,color='black',linestyle='dotted')

    ax1.set_ylabel(' (arb. units)', fontsize=10)
    ax1.tick_params(axis='x', labelsize='10' )
    ax1.tick_params(axis='y', labelsize='10' )
    # ax1.set_xlim([-200,-150])
    # ax1.set_ylim([-0.01,0.01])
    # ax1.set_yticks([-1,-0.5,0,0.5,1]) # 设置刻度
    ax1.set_xticklabels([])

plt.show()
