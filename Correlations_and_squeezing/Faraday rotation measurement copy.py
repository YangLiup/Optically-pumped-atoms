# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2024年03月07日
"""
# -*- coding:utf-8 -*-
import sys
sys.path.append(r"D:\python\pythonProject\Optically_pumped_atoms\my_functions")
import numpy as np
import matplotlib.pyplot as plt
from Generate_a_squeezed_state_by_QND import Generate_a_squeezed_state_by_QND
from alkali_atom_uncoupled_to_coupled import alkali_atom_uncoupled_to_coupled
from scipy.linalg import *
from spin_operators_of_2or1_alkali_metal_atoms import spin_operators_of_2or1_alkali_metal_atoms
from ptr import ptr
from scipy.linalg import *
import scienceplots
from matplotlib.ticker import FuncFormatter
from scipy.stats import norm
import pandas as pd

# ax, ay, az, bx, by, bz, a1x, a2x, a1y, a2y, a1z, a2z, Fx, Fy, Fz
x = np.array(pd.read_csv('D:/Optically-pumped-atoms/data/x.csv')).flatten()
C_1z2z = np.array(pd.read_csv('D:/Optically-pumped-atoms/data/C_1z2z.csv')).flatten()
C_1z1z  = np.array(pd.read_csv('D:/Optically-pumped-atoms/data/C_1z1z.csv')).flatten()
T=10
dt=0.01
Jx=np.arange(-4,4,0.01)
plt.style.use(['science'])
with plt.style.context(['science']):
    tt = np.arange(0, T, dt)/10
    fig = plt.figure(figsize=(3.35, 4.8))
    ax_1  = fig.add_subplot(212)
    w=3.5*np.sqrt(5)/500**(1/3)
    ax_1.hist(x,bins=[-2.7,-2.1,-1.5, -0.9, -0.3, 0.3, 0.9,1.5,2.1,2.7],density=True,edgecolor='white', alpha=0.8)
    ax_1.plot([],[])
    ax_1.plot([],[])
    ax_1.plot([],[])
    ax_1.plot([],[])
    ax_1.set_ylim([0,0.6])
#     ax_1.plot(Jx,1/np.sqrt(2*np.pi)*np.exp(-Jx**2/2),linestyle='dashed')
    ax_1.set_xlabel('$\langle{ F_x}\\rangle^{\prime}$', fontsize=9)
    ax_1.set_ylabel('Distribution@$t=1$', fontsize=9)
    ax_1.tick_params(axis='x', labelsize='9' )
    ax_1.tick_params(axis='y', labelsize='9' )
    ax_1.text(2.5, 0.55, '(b)',fontsize=8)

    ax_2  = fig.add_subplot(211)
    down=C_1z1z-np.sqrt(C_1z2z)/2
    up=C_1z1z+np.sqrt(C_1z2z)/2
    ax_2.plot(tt,C_1z1z)
    ax_2.fill_between(tt, down,up,facecolor = 'red', alpha = 0.5)
    ax_2.text(8.5/10, 0.1, '$\mathbb{E}[\langle{ F_x}\\rangle^{\prime}]$',fontsize=6)
    ax_2.annotate('$\sqrt{ {\mathrm{var}} ({F_x}) }$', xy=(0.3/10, 1), xytext=(2/10, 1),
            arrowprops=dict(arrowstyle='->', color='red'),fontsize=6)

    ax_2.plot(tt, np.zeros(1000),color='black',linestyle='dashed')
    ax_2.set_xlabel('t ($1/\Phi$)', fontsize=9)
    ax_2.set_ylabel('$\langle{ F_x}\\rangle^{\prime}$', fontsize=9)
    ax_2.tick_params(axis='x', labelsize='9' )
    ax_2.tick_params(axis='y', labelsize='9' )
    ax_2.text(0.975, 1.5, '(a)',fontsize=8)
    # ax_2.plot(tt, np.zeros(n),color='black',linestyle='dashed',linewidth='0.5')
    # ax_2.set_xlabel('t ($1/\Phi$)', fontsize=10)
    # ax_2.set_ylabel('$\\tilde{J_x}$', fontsize=10)
    # ax_2.tick_params(axis='x', labelsize='10' )
    # ax_2.tick_params(axis='y', labelsize='10' )
    plt.savefig('squeezing.png', dpi=1000)

    plt.show()
