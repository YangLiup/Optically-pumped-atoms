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
    ax_2  = fig.add_subplot(211)
    ax_2.plot(tt,C_1z1z,linewidth='1.5')
    ax_2.set_xlabel('Time', fontsize=19,fontweight='bold')
    ax_2.set_ylabel('Collective spin', fontsize=19,fontweight='bold')
    ax_2.tick_params(axis='x', labelsize='9' )
    ax_2.tick_params(axis='y', labelsize='9' )
    ax_2.set_title('Spin noise signal', fontsize=20,fontweight='bold')
    ax_2.set_xticklabels([])
    ax_2.set_yticklabels([])
    plt.savefig('squeezing.png', dpi=1000)

    plt.show()
