# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2024年03月07日
"""
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from scipy.special import wofz
import matplotlib.ticker as ticker

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
    delta_nva2=np.arange(-10,10,0.01)
    fig = plt.figure(figsize=(3.35, 6))
    plt.rc('font',family='Times New Roman')
    ax1= fig.add_subplot(311)
    # ax1.plot([],[])
    # ax1.plot([],[])
    p1,=ax1.plot(delta_nva2,chip(delta_nva2,0)/np.max(chip(delta_nva2,0)),linewidth=0.6)
    p2,=ax1.plot(delta_nva2,chim(delta_nva2,0)/np.max(chip(delta_nva2,0)),linewidth=0.6)
    ax1.plot(delta_nva2,np.zeros(len(delta_nva2)),linewidth=0.2,color='black',linestyle='dashed')

    ax1.set_ylabel(' (arb. units)', fontsize=9)
    ax1.tick_params(axis='x', labelsize='9' )
    ax1.tick_params(axis='y', labelsize='9' )
    ax1.set_yticks([-1,-0.5,0,0.5,1]) # 设置刻度
    ax1.set_xticklabels([])

    ax1.text(-10, 0.8, '(a) P=0',fontsize=8)

    ax2 = fig.add_subplot(312)
    # ax2.plot([],[])
    # ax2.plot([],[])
    ax2.plot(delta_nva2,chip(delta_nva2,0.5)/np.max(chip(delta_nva2,0)),linewidth=0.6)
    ax2.plot(delta_nva2,chim(delta_nva2,0.5)/np.max(chip(delta_nva2,0)),linewidth=0.6)
    ax2.plot(delta_nva2,np.zeros(len(delta_nva2)),linewidth=0.2,color='black',linestyle='dashed')
    ax2.set_ylabel(' (arb. units)', fontsize=9)
    ax2.tick_params(axis='x', labelsize='9' )
    ax2.tick_params(axis='y', labelsize='9' )
    ax2.set_xticklabels([],fontsize='small')
    ax2.set_yticks([-1,-0.5,0,0.5,1]) # 设置刻度


    ax2.text(-10, 0.8, '(b) P=0.5',fontsize=8)
    ax2.text(4.8, -0.9, '$1\\to2$',fontsize=6)
    ax2.text(-2.0, -0.9, '$2\\to2$',fontsize=6)
    ax2.text(-5.5, -0.9, '$2\\to1$',fontsize=6)
    ax2.text(1.6, -0.9, '$1\\to1$',fontsize=6)

    # plt.scatter(4.5,0,s=1.5,c='blue')
    # plt.scatter(-2.3,0,s=1.5,c='purple')
    # plt.scatter(-3.1,0,s=1.5,c='green')
    # plt.scatter(3.7,0,s=1.5,c='olive')
    plt.vlines(4.5, -1, 1, linestyles ="dashed", colors ="k",linewidth=0.4)
    plt.vlines(-2.3, -1, 1, linestyles ="dashed", colors ="k",linewidth=0.4)
    plt.vlines(-3.1, -1, 1, linestyles ="dashed", colors ="k",linewidth=0.4)
    plt.vlines(3.7, -1, 1, linestyles ="dashed", colors ="k",linewidth=0.4)



    ax3 = fig.add_subplot(313)
    # ax3.plot([],[])
    # ax3.plot([],[])
    ax3.plot(delta_nva2,chip(delta_nva2,0.99)/np.max(chip(delta_nva2,0)),linewidth=0.6)
    ax3.plot(delta_nva2,chim(delta_nva2,0.99)/np.max(chip(delta_nva2,0)),linewidth=0.6)
    ax3.plot(delta_nva2,np.zeros(len(delta_nva2)),linewidth=0.2,color='black',linestyle='dashed')

    ax3.set_ylabel(' (arb. units)', fontsize=9)
    ax3.tick_params(axis='x', labelsize='9' )
    ax3.tick_params(axis='y', labelsize='9' )
    # ax3.set_yticks([-1,-0.5,0,0.5,1]) # 设置刻度
    ax3.text(-10, 0.8, '(c) P=0.99',fontsize=8)
    ax3.set_yticks([-1,-0.5,0,0.5,1]) # 设置刻度


  


    ax1.legend([p1,p2],["$\chi_+$", "$\chi_-$"], loc='upper right',prop={'size':8})
    # plt.xlabel('Frequency (Hz)', fontsize=12)
    # plt.ylabel(' PSD ($ N \chi_+^2$/Hz)', fontsize=12)
    plt.xlabel('$\\nu-\\nu_0$ (GHz)', fontsize=9)

    plt.savefig('myself/imag/detuning.png', dpi=1000)
plt.show()
