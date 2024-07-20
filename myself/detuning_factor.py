# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2024年03月07日
"""
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from scipy.special import wofz
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
    ax1.plot([],[])
    # ax1.plot([],[])
    y_1=chip(delta_nva2,0)/np.max(chip(delta_nva2,0))
    y_2=chim(delta_nva2,0)/np.max(chip(delta_nva2,0))
    p1,=ax1.plot(delta_nva2,y_1)
    p2,=ax1.plot(delta_nva2,y_2)
    ax1.plot(delta_nva2,np.zeros(len(delta_nva2)),linewidth=0.2,color='black',linestyle='dashed')

    ax1.set_ylabel(' (arb. units)', fontsize=10)
    ax1.tick_params(axis='x', labelsize='10' )
    ax1.tick_params(axis='y', labelsize='10' )
    ax1.set_yticks([-1,-0.5,0,0.5,1]) # 设置刻度
    ax1.set_xticklabels([])

    ax1.text(-10, 0.8, '(a) P=0',fontsize=8)

    #嵌入绘制局部放大图的坐标系
    axins = ax1.inset_axes((0.7, 0.1, 0.25, 0.2))

    #在子坐标系中绘制原始数据
    axins.plot([], [])
    axins.plot(delta_nva2, y_1)
    axins.plot(delta_nva2, y_2)
    axins.tick_params(axis='x', labelsize='6' )
    axins.tick_params(axis='y', labelsize='6' )

    #设置放大区间
    zone_left = 1650
    zone_right = 1750

    #坐标轴的扩展比例（根据实际数据调整）
    x_ratio = 0.5 # x轴显示范围的扩展比例
    y_ratio = 0.5 # y轴显示范围的扩展比例

    #X轴的显示范围
    xlim0 = delta_nva2[zone_left]-(delta_nva2[zone_right]-delta_nva2[zone_left])*x_ratio
    xlim1 = delta_nva2[zone_right]+(delta_nva2[zone_right]-delta_nva2[zone_left])*x_ratio

    #Y轴的显示范围
    y = np.hstack((y_1[zone_left:zone_right], y_2[zone_left:zone_right]))
    ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
    ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio

    #调整子坐标系的显示范围
    axins.set_xlim(xlim0, xlim1)
    axins.set_ylim(ylim0, ylim1)

    #建立父坐标系与子坐标系的连接线
    #loc1 loc2: 坐标系的四个角
    #1 (右上) 2 (左上) 3(左下) 4(右下)
    mark_inset(ax1, axins, loc1=3, loc2=1, fc="none", ec='k', lw=0.2)

    ax2 = fig.add_subplot(312)
    ax2.plot([],[])
    # ax2.plot([],[])
    ax2.plot(delta_nva2,chip(delta_nva2,0.5)/np.max(chip(delta_nva2,0)))
    ax2.plot(delta_nva2,chim(delta_nva2,0.5)/np.max(chip(delta_nva2,0)))
    ax2.plot(delta_nva2,np.zeros(len(delta_nva2)),linewidth=0.2,color='black',linestyle='dashed')
    ax2.set_ylabel(' (arb. units)', fontsize=10)
    ax2.tick_params(axis='x', labelsize='10' )
    ax2.tick_params(axis='y', labelsize='10' )
    ax2.set_xticklabels([])
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
    ax3.plot([],[])
    # ax3.plot([],[])
    ax3.plot(delta_nva2,chip(delta_nva2,0.99)/np.max(chip(delta_nva2,0)))
    ax3.plot(delta_nva2,chim(delta_nva2,0.99)/np.max(chip(delta_nva2,0)))
    ax3.plot(delta_nva2,np.zeros(len(delta_nva2)),linewidth=0.2,color='black',linestyle='dashed')

    ax3.set_ylabel(' (arb. units)', fontsize=10)
    ax3.tick_params(axis='x', labelsize='10' )
    ax3.tick_params(axis='y', labelsize='10' )
    # ax3.set_yticks([-1,-0.5,0,0.5,1]) # 设置刻度
    ax3.text(-10, 1, '(c) P=0.99',fontsize=8)
    ax3.set_yticks([-1,-0.5,0,0.5,1]) # 设置刻度
    ax1.legend([p1,p2],["$\chi_+$", "$\chi_-$"], loc='upper right',prop={'size':9})
    # plt.xlabel('Frequency (Hz)', fontsize=12)
    # plt.ylabel(' PSD ($ N \chi_+^2$/Hz)', fontsize=12)
    plt.xlabel('$\\nu-\\nu_0$ (GHz)', fontsize=10)

    plt.savefig('myself/imag/detuning.png', dpi=1000)
plt.show()
