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
import matplotlib
matplotlib.use('TkAgg')  # 或者 'Qt5Agg' 等其他后端

t=np.arange(0,8*np.pi,0.01)
y1=np.sin(t/2)**2
y2=1-y1

plt.style.use(['science'])
with plt.style.context(['science']):
    plt.rc('font',family='Times New Roman')
    plt.plot(t,y1)
    plt.plot(t,y2)
    plt.savefig('rabi.png', dpi=600)

plt.show()





