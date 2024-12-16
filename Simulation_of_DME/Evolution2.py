import sys
sys.path.append(r"D:\python\pythonProject\Optically_pumped_atoms\my_functions")

import matplotlib.pyplot as plt
from spin_operators_of_2or1_alkali_metal_atoms import spin_operators_of_2or1_alkali_metal_atoms
from alkali_atom_uncoupled_to_coupled import alkali_atom_uncoupled_to_coupled
from qutip import *
from Master_equation import master_equation
from scipy.linalg import *
import numpy as np
from matplotlib import ticker
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

T1=8000
T2=5000
T3=4000


dt=0.01
Px1,Pz1,transverse1,longitude1=master_equation(3/2,1,0.01,T1,10)
Px2,Pz2,transverse2,longitude2=master_equation(3/2,1,0.05,T2,10)
Px3,Pz3,transverse3,longitude3=master_equation(3/2,1,0.25,T3,10)


tt1=np.arange(0,T1,dt)
tt2=np.arange(0,T2,dt)
tt3=np.arange(0,T3,dt)

plt.style.use(['science'])
with plt.style.context(['science']):
    plt.rc('font',family='Times New Roman')
    fig = plt.figure(figsize=(3.2, 5))
    # plt.ylabel('Polarization ', fontsize=8)
    # plt.xlabel('$t$ ($1/R_{\mathrm{se}}$)', fontsize=8)
    # plt.xticks(ticks=[])
    # plt.yticks(ticks=[])
    ax1 = fig.add_subplot(311)
    ax1.plot(tt1, Px1)
    ax1.plot(tt1, Pz1)
    ax1.plot(tt1, transverse1)
    ax1.plot(tt1, longitude1)
    # ax1.set_xlim([0,50000])
    # ax1.set_ylim([0,1])

    ax1.set_ylabel('Polarization', fontsize=8)
    # ax1.axes.xaxis.set_ticklabels([])
    ax1.text(7200, 0.63, '(a)',fontsize=8)
    ax1.tick_params(axis='both', which='major', labelsize=8)
    ax1.tick_params(axis='both', which='minor', labelsize=8)

    ax2 = fig.add_subplot(312)
    ax2.plot(tt2, Px2)
    ax2.plot(tt2, Pz2)
    ax2.plot(tt2, transverse2)
    ax2.plot(tt2, longitude2)

    # ax2.set_xlabel('$t$ ($1/R_{\mathrm{se}}$)', fontsize=9)
    ax2.set_ylabel('Polarization ', fontsize=8)
    # ax2.set_xlim([0,10000])
    ax2.text(4500, 0.63, '(b)',fontsize=8)
    # ax2.axes.xaxis.set_ticklabels([])
    ax1.tick_params(axis='both', which='major', labelsize=8)
    ax1.tick_params(axis='both', which='minor', labelsize=8)

    ax2.tick_params(axis='both', which='major', labelsize=8)
    ax2.tick_params(axis='both', which='minor', labelsize=8)

    ax3 = fig.add_subplot(313)
    ax3.plot(tt3, Px3)
    ax3.plot(tt3, Pz3)
    ax3.plot(tt3, transverse3)
    ax3.plot(tt3, longitude3)

    # ax3.set_xlabel('$t$ ($1/R_{\mathrm{se}}$)', fontsize=8)
    ax3.set_ylabel('Polarization ', fontsize=8)
    # ax3.set_xlim([0,2000])
    ax3.text(3600, 0.63, '(c)', fontsize=8)

    ax3.tick_params(axis='both', which='major', labelsize=8)
    ax3.tick_params(axis='both', which='minor', labelsize=8)
    ax3.set_xlabel('$t$ ($1/R_{\mathrm{se}}$)', fontsize=8)
    ax3.legend(["$P_{\perp}^{\mathrm{DME}}$", "$P_z^{\mathrm{DME}}$", "$P_{\perp}^{\mathrm{EBE}}$", "$P_z^{\mathrm{EBE}}$"],
               loc='center right', prop={'size': 7})
    

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True) 
    formatter.set_powerlimits((0,0)) 
    ax3.xaxis.set_major_formatter(formatter)
    ax2.xaxis.set_major_formatter(formatter)
    ax1.xaxis.set_major_formatter(formatter)
    # ax2.axes.yaxis.set_ticklabels([])
    # ax4.axes.yaxis.set_ticklabels([])


    # 嵌入绘制局部放大图的坐标系
    axins = inset_axes(ax3, width="40%", height="30%",loc='lower left',
                    bbox_to_anchor=(0.3, 0.58, 0.8, 1),
                    bbox_transform=ax3.transAxes)

    # 在子坐标系中绘制原始数据
    axins.plot(tt3, Px3, linewidth=1)
    axins.plot([], [], linewidth=1)
    axins.plot(tt3, transverse3,  linewidth=1)
    axins.tick_params(axis='both', which='major', labelsize=6)
    axins.tick_params(axis='both', which='minor', labelsize=6)

    # 设置放大区间
    zone_left = 300*100
    zone_right = 500*100

    # 坐标轴的扩展比例（根据实际数据调整）
    x_ratio = 0.5 # x轴显示范围的扩展比例
    y_ratio = 0.5 # y轴显示范围的扩展比例

    # X轴的显示范围
    xlim0 = tt3[zone_left]-(tt3[zone_right]-tt3[zone_left])*x_ratio
    xlim1 = tt3[zone_right]+(tt3[zone_right]-tt3[zone_left])*x_ratio

    # Y轴的显示范围
    y = np.hstack((transverse3[zone_left:zone_right], Px3[zone_left:zone_right]))
    ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
    ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio

    # 调整子坐标系的显示范围
    axins.set_xlim(xlim0, xlim1)
    axins.set_ylim(ylim0, ylim1)

    # 建立父坐标系与子坐标系的连接线
    # loc1 loc2: 坐标系的四个角
    # 1 (右上) 2 (左上) 3(左下) 4(右下)
    # mark_inset(ax3, axins, loc1=3, loc2=4, lw=0.5,linestyle='--')


    plt.tight_layout()
    plt.savefig('Evolution.png', dpi=1000)
plt.show()