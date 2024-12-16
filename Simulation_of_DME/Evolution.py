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


T1=100000
T2=8000
T3=800


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
    ax1.text(90000, 0.65, '(a)',fontsize=8)
    ax1.tick_params(axis='both', which='major', labelsize=8)
    ax1.tick_params(axis='both', which='minor', labelsize=8)

    ax2 = fig.add_subplot(312)
    ax2.plot(tt2, Px2)
    ax2.plot(tt2, Pz2)
    ax2.plot(tt2, transverse2)
    ax2.plot(tt2, longitude2)
    ax2.legend(["$P_{\perp}^{\mathrm{DME}}$", "$P_z^{\mathrm{DME}}$", "$P_{\perp}^{\mathrm{EBE}}$", "$P_z^{\mathrm{EBE}}$"],
               loc='lower right', prop={'size': 8})

    # ax2.set_xlabel('$t$ ($1/R_{\mathrm{se}}$)', fontsize=9)
    ax2.set_ylabel('Polarization ', fontsize=8)
    # ax2.set_xlim([0,10000])
    ax2.text(7200, 0.65, '(b)',fontsize=8)
    
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
    ax3.text(720, 0.65, '(c)', fontsize=8)

    ax3.tick_params(axis='both', which='major', labelsize=8)
    ax3.tick_params(axis='both', which='minor', labelsize=8)
    ax3.set_xlabel('$t$ ($1/R_{\mathrm{se}}$)', fontsize=8)
    

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True) 
    formatter.set_powerlimits((0,0)) 
    ax3.xaxis.set_major_formatter(formatter)
    ax2.xaxis.set_major_formatter(formatter)
    ax1.xaxis.set_major_formatter(formatter)
    # ax2.axes.yaxis.set_ticklabels([])
    # ax4.axes.yaxis.set_ticklabels([])
    plt.tight_layout()
    plt.savefig('Evolution.png', dpi=1000)
plt.show()