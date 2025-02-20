# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2023年12月24日
"""

import sys
sys.path.append(r"/Users/liyang/Documents/GitHub/Optically_polarized_atoms/my_functions")
sys.path.append(r"D:\Software\python\pythonProject\Optically-pumped-atoms\Optically-pumped-atoms\my_functions")
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import mpl_toolkits.axisartist as axisartist


t=np.arange(0,4*np.pi,0.01)
y1=np.sin(t/2)**2
y2=1-y1

plt.style.use(['science'])
with plt.style.context(['science']):
    plt.rc('font',family='Times New Roman')
    #通过set_visible方法设置绘图区所有坐标轴隐藏
    fig=plt.figure()
    #使用axisartist.Subplot方法创建一个绘图区对象ax
    ax = axisartist.Subplot(fig, 111)  
    #将绘图区对象添加到画布中
    fig.add_axes(ax)
    ax.axis[:].set_visible(False)
    #ax.new_floating_axis代表添加新的坐标轴
    ax.axis["x"] = ax.new_floating_axis(0,0)
    #给x坐标轴加上箭头
    ax.axis["x"].set_axisline_style("->", size = 1.0)
    #添加y坐标轴，且加上箭头
    ax.axis["y"] = ax.new_floating_axis(1,0)
    ax.axis["y"].set_axisline_style("-|>", size = 1.0)
    #设置x、y轴上刻度显示方向
    ax.axis["x"].set_axis_direction("bottom")
    ax.axis["y"].set_axis_direction("left")

    p1,=ax.plot(t,y1,color='orangered')
    p2,=ax.plot(t,y2,color='orangered',linestyle='dashed')
    ax.axis["x"].set_label('$\Omega t$')
    ax.axis["y"].set_label('$P$')

    ax.set_xticks([0, 2*np.pi,4*np.pi])
    ax.set_yticks([0,0.5, 1])
    ax.set_xticklabels(['$0$', '$2\pi$', '$4\pi$'], rotation=0, fontsize=12)
    ax.legend([p1, p2],["$ P_g$", "$ P_e$"],loc='upper right',ncol=2)
    plt.ylim([0,1.2])
    plt.savefig('rabi.png', dpi=600)
plt.show()






