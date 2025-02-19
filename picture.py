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


t=np.arange(0,6*np.pi,0.01)
y1=np.sin(t/2)**2
y2=1-y1

plt.style.use(['science'])
with plt.style.context(['science']):
    plt.rc('font',family='Times New Roman')
    fig=plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    p1,=ax.plot(t,y1,color='black')
    p2,=ax.plot(t,y2,color='black',linestyle='dashed')
    plt.xlabel('$\Omega t$')
    plt.ylabel('$P$')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, labeltop=False)
    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True, labelright=False)
    ax.set_xticks([0, 2*np.pi,4*np.pi, 6*np.pi])
    ax.set_yticks([0,0.5, 1])
    ax.set_xticklabels(['$0$', '$2\pi$', '$4\pi$', '$6\pi$'], rotation=0, fontsize=12)
    ax.legend([p1, p2],["$ P_g$", "$ P_e$"],loc='upper right')
    plt.ylim([0,1.3])
    plt.savefig('rabi.png', dpi=600)







