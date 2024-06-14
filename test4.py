import matplotlib.pyplot as plt
from my_functions.spin_operators_of_2or1_alkali_metal_atoms import spin_operators_of_2or1_alkali_metal_atoms
from my_functions.alkali_atom_uncoupled_to_coupled import alkali_atom_uncoupled_to_coupled
from qutip import *
from sympy.physics.quantum.spin import Rotation
from sympy import pi
from test5 import master_equation
from scipy.linalg import *
import numpy as np
import scienceplots
T1=20000
dt=0.01
T2=1000
T3=1000
T4=1000
tp1=10
tp2=2
tp3=2
tp4=2
transverse1, longitude1, Pz1, Px1=master_equation(T1, dt, tp1,0.01)
transverse2, longitude2, Pz2, Px2=master_equation(T2, dt, tp2,0.1)
transverse3, longitude3, Pz3, Px3=master_equation(T3, dt, tp3,0.2)
transverse4, longitude4, Pz4, Px4=master_equation(T4, dt, tp4,1)
plt.style.use(['science','nature'])
with plt.style.context(['science','nature']):
    fig = plt.figure(figsize=(3.2,8))
    ax1 = fig.add_subplot(411)
    ax1.plot(np.arange(0,T1,dt), Px1 )
    ax1.plot(np.arange(0,T1,dt), Pz1  )
    ax1.plot(np.arange(0,T1,dt), transverse1)
    ax1.plot(np.arange(0,T1,dt), longitude1)
    ax1.set_ylabel('Polarization', fontsize=11)
    ax1.tick_params(axis='x', labelsize='10' )
    ax1.tick_params(axis='y', labelsize='10' )

    # ax1.set_ylabel('$\Gamma^+\;(R_{\\rm{se}})$',fontsize='12')
    # ax1.set_ylim([0,0.05])
    ax1.set_xlim([0,20000])
    # ax1.set_xticklabels([])
    ax2 = fig.add_subplot(412)
    ax2.plot(np.arange(0,T2,dt), Px2 )
    ax2.plot(np.arange(0,T2,dt), Pz2  )
    ax2.plot(np.arange(0,T2,dt), transverse2)
    ax2.plot(np.arange(0,T2,dt), longitude2)
    ax2.set_ylabel('Polarization', fontsize=11)
    ax2.tick_params(axis='x', labelsize='10' )
    ax2.tick_params(axis='y', labelsize='10' )
    ax2.set_xlim([0,1000])
    ax3 = fig.add_subplot(413)
    ax3.plot(np.arange(0,T2,dt), Px3 )
    ax3.plot(np.arange(0,T2,dt), Pz3  )
    ax3.plot(np.arange(0,T2,dt), transverse3)
    ax3.plot(np.arange(0,T2,dt), longitude3)
    ax3.set_ylabel('Polarization', fontsize=11)
    ax3.tick_params(axis='x', labelsize='10' )
    ax3.tick_params(axis='y', labelsize='10' )
    ax3.set_xlim([0,1000])
    tt=np.arange(0,T2,dt)
    ax4 = fig.add_subplot(414)
    ax4.plot(np.arange(0,T2,dt), Px4 )
    ax4.plot(np.arange(0,T2,dt), Pz4  )
    ax4.plot(np.arange(0,T2,dt), transverse4)
    ax4.plot(np.arange(0,T2,dt), longitude4)
    ax4.set_ylabel('Polarization', fontsize=11)
    ax4.set_xlabel('Time $(1/R_{\\rm{se}})$', fontsize=11)
    ax4.tick_params(axis='x', labelsize='10' )
    ax4.tick_params(axis='y', labelsize='10' )
    ax4.set_xlim([0,1000])
    ax4.legend( ["$P_x^{\\rm{NB}}$", "$P_z^{\\rm{NB}}$", "$P_x^{\\rm{DM}}$","$P_z^{\\rm{DM}}$"],
               loc='center right', prop={'size': 10})
    
    axins = ax4.inset_axes((0.2, 0.2, 0.4, 0.3))
    axins.plot(tt,  Px4)
    axins.plot([], [])
    axins.plot(tt, transverse4)
    # 设置放大区间
    zone_left = 10*100*45
    zone_right = 10*100*55

    # 坐标轴的扩展比例（根据实际数据调整）
    x_ratio = 0.5 # x轴显示范围的扩展比例
    y_ratio = 0.2 # y轴显示范围的扩展比例

    # X轴的显示范围
    xlim0 = tt[zone_left]-(tt[zone_right]-tt[zone_left])*x_ratio
    xlim1 = tt[zone_right]+(tt[zone_right]-tt[zone_left])*x_ratio

    # Y轴的显示范围
    y = np.hstack((Pz4[zone_left:zone_right], transverse4[zone_left:zone_right]))
    ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
    ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio

    # 调整子坐标系的显示范围
    axins.set_xlim(xlim0, xlim1)
    axins.set_ylim(-0.025, 0.025)
 
    # ax2.set_ylim([0,0.8])
    # ax2.set_xlim([0,1])
    # ax1.xticks(fontsize=10)
    # ax1.yticks(fontsize=10)
    # ax2.xticks(fontsize=10)
    # ax2.yticks(fontsize=10)
    

    # plt.xlim(0, 200)
    # plt.ylim(0, 18)


    plt.savefig('imag/Evolution1.png', dpi=600)
plt.show()